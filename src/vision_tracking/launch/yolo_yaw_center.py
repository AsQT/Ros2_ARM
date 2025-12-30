#!/usr/bin/env python3
import os
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from ultralytics import YOLO


# ---------------- utils ----------------

def clamp_angle_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return float(a)

def map_to_pm90(a: float) -> float:
    a = clamp_angle_deg(a)
    if a >= 90.0:
        a -= 180.0
    if a < -90.0:
        a += 180.0
    return float(a)

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    pts = pts[idx]
    # rotate so first is top-left-ish
    s = pts.sum(axis=1)
    k = int(np.argmin(s))
    pts = np.roll(pts, -k, axis=0)
    return pts

def poly_area(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=np.float32)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def gamma_lut(gamma: float) -> np.ndarray:
    gamma = max(0.05, float(gamma))
    inv = 1.0 / gamma
    return np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8)

def draw_arrow(img: np.ndarray, center: Tuple[int, int], yaw_deg: float, length: int = 70):
    cx, cy = center
    ang = math.radians(yaw_deg)
    x2 = int(round(cx + length * math.cos(ang)))
    y2 = int(round(cy + length * math.sin(ang)))
    cv2.arrowedLine(img, (cx, cy), (x2, y2), (255, 0, 0), 3, tipLength=0.25)

def rosimg_to_bgr(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    step = int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        buf = np.pad(buf, (0, h * step - buf.size), mode="constant")
    row = buf.reshape((h, step))

    if enc in ("rgb8", "bgr8"):
        img = row[:, :w * 3].reshape((h, w, 3))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    if enc in ("mono8", "8uc1"):
        g = row[:, :w].reshape((h, w))
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # fallback assume bgr8
    img = row[:, :w * 3].reshape((h, w, 3))
    return img.copy()

def bgr_to_rosimg(img_bgr: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height = int(img_bgr.shape[0])
    msg.width = int(img_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(msg.width * 3)
    msg.data = img_bgr.tobytes()
    return msg

# -------------- line model --------------

@dataclass
class Line:
    # normalized ax + by + c = 0 (sqrt(a^2+b^2)=1)
    a: float
    b: float
    c: float
    theta: float   # direction angle in [0, pi)
    length: float

def segment_to_line(x1, y1, x2, y2) -> Optional[Line]:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return None

    # direction angle (mod pi)
    theta = math.atan2(dy, dx)
    if theta < 0:
        theta += math.pi
    if theta >= math.pi:
        theta -= math.pi

    # line coefficients from two points
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    n = math.hypot(a, b)
    if n < 1e-6:
        return None
    a /= n
    b /= n
    c /= n
    return Line(a=a, b=b, c=c, theta=theta, length=L)

def intersect(L1: Line, L2: Line) -> Optional[Tuple[float, float]]:
    det = L1.a * L2.b - L2.a * L1.b
    if abs(det) < 1e-6:
        return None
    x = (L1.b * L2.c - L2.b * L1.c) / det
    y = (L2.a * L1.c - L1.a * L2.c) / det
    return (float(x), float(y))

def unify_normal(line: Line, nref: Tuple[float, float]) -> Line:
    # flip (a,b,c) so that (a,b) dot nref >= 0
    dot = line.a * nref[0] + line.b * nref[1]
    if dot >= 0:
        return line
    return Line(a=-line.a, b=-line.b, c=-line.c, theta=line.theta, length=line.length)

# --------- cluster 2 directions (double-angle) ---------

def kmeans2_directions(lines: List[Line]) -> Optional[Tuple[float, float]]:
    """
    Returns 2 center directions theta1, theta2 (in [0,pi)).
    Uses double-angle unit vectors to handle 180 symmetry.
    """
    if len(lines) < 2:
        return None

    # vectors on unit circle for double-angle
    V = []
    W = []
    for ln in lines:
        V.append([math.cos(2.0 * ln.theta), math.sin(2.0 * ln.theta)])
        W.append(ln.length)
    V = np.asarray(V, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32) + 1e-6

    # init centers: longest and farthest (min dot)
    i0 = int(np.argmax(W))
    c1 = V[i0].copy()
    dots = V @ c1
    i1 = int(np.argmin(dots))
    c2 = V[i1].copy()

    # normalize
    c1 /= (np.linalg.norm(c1) + 1e-6)
    c2 /= (np.linalg.norm(c2) + 1e-6)

    for _ in range(6):
        d1 = V @ c1
        d2 = V @ c2
        lab = d1 >= d2

        if lab.all() or (~lab).all():
            break

        w1 = W[lab]
        v1 = V[lab]
        w2 = W[~lab]
        v2 = V[~lab]

        c1 = (v1 * w1[:, None]).sum(axis=0)
        c2 = (v2 * w2[:, None]).sum(axis=0)
        c1 /= (np.linalg.norm(c1) + 1e-6)
        c2 /= (np.linalg.norm(c2) + 1e-6)

    # convert back from double-angle vector -> theta
    th1 = 0.5 * math.atan2(float(c1[1]), float(c1[0]))
    th2 = 0.5 * math.atan2(float(c2[1]), float(c2[0]))
    if th1 < 0:
        th1 += math.pi / 2.0
    if th2 < 0:
        th2 += math.pi / 2.0
    # ensure [0,pi)
    th1 = th1 % math.pi
    th2 = th2 % math.pi
    return (float(th1), float(th2))

def assign_group(theta: float, th1: float, th2: float) -> int:
    # compare by double-angle dot
    v = np.array([math.cos(2 * theta), math.sin(2 * theta)], dtype=np.float32)
    c1 = np.array([math.cos(2 * th1), math.sin(2 * th1)], dtype=np.float32)
    c2 = np.array([math.cos(2 * th2), math.sin(2 * th2)], dtype=np.float32)
    return 0 if float(v @ c1) >= float(v @ c2) else 1

# --------- build largest parallelogram from lines ---------

def best_parallelogram_from_lines(lines: List[Line], roi_w: int, roi_h: int) -> Optional[np.ndarray]:
    """
    Pick 2 dominant directions, pick 2 parallel lines per direction (extremes),
    intersect to form parallelogram, choose max area among candidates.
    Returns 4 corners in ROI coords (4,2) float32.
    """
    if len(lines) < 4:
        return None

    centers = kmeans2_directions(lines)
    if centers is None:
        return None
    th1, th2 = centers

    g0 = []
    g1 = []
    for ln in lines:
        (g0 if assign_group(ln.theta, th1, th2) == 0 else g1).append(ln)

    if len(g0) < 2 or len(g1) < 2:
        return None

    def group_candidates(g: List[Line], th_center: float, topN: int = 18, M: int = 3):
        # sort by length desc
        g = sorted(g, key=lambda z: z.length, reverse=True)[:topN]

        # group normal from direction th_center
        nx = -math.sin(th_center)
        ny =  math.cos(th_center)
        nref = (nx, ny)

        gg = [unify_normal(ln, nref) for ln in g]
        ds = np.array([-ln.c for ln in gg], dtype=np.float32)  # signed distance since normed
        idx = np.argsort(ds)

        small = [gg[i] for i in idx[:M]]
        large = [gg[i] for i in idx[-M:]]
        pairs = []
        for a in small:
            for b in large:
                if abs((-a.c) - (-b.c)) < 2.0:
                    continue
                pairs.append((a, b))
        return pairs

    pairs0 = group_candidates(g0, th1)
    pairs1 = group_candidates(g1, th2)
    if not pairs0 or not pairs1:
        return None

    best_poly = None
    best_area = 0.0

    # brute combine few candidates
    for (A1, A2) in pairs0:
        for (B1, B2) in pairs1:
            p11 = intersect(A1, B1)
            p12 = intersect(A1, B2)
            p21 = intersect(A2, B1)
            p22 = intersect(A2, B2)
            if p11 is None or p12 is None or p21 is None or p22 is None:
                continue

            poly = np.array([p11, p12, p22, p21], dtype=np.float32)  # roughly a quad
            poly = order_points_clockwise(poly)

            area = poly_area(poly)
            if area < 100.0:
                continue

            # inside ROI (allow small margin)
            m = 8.0
            if (poly[:, 0].min() < -m or poly[:, 1].min() < -m or
                poly[:, 0].max() > (roi_w - 1 + m) or poly[:, 1].max() > (roi_h - 1 + m)):
                continue

            # avoid super skinny
            e = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
            if float(e.min()) < 12.0:
                continue

            if area > best_area:
                best_area = area
                best_poly = poly

    return best_poly

# ---------------- main node ----------------

class YoloHoughParallelogram(Node):
    def __init__(self):
        super().__init__("yolo_hough_parallelogram")

        # params
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("model_path", "")
        self.declare_parameter("target_class", "wood")
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("iou", 0.4)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_det", 30)

        self.declare_parameter("gamma", 1.6)
        self.declare_parameter("morph_kernel", 5)

        self.declare_parameter("canny1", 50)
        self.declare_parameter("canny2", 150)
        self.declare_parameter("hough_threshold", 40)
        self.declare_parameter("min_line_length", 25)
        self.declare_parameter("max_line_gap", 10)

        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("draw_hough_lines", False)

        # load
        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.target_class = self.get_parameter("target_class").value
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.max_det = int(self.get_parameter("max_det").value)

        self.gamma = float(self.get_parameter("gamma").value)
        self.morph_k = int(self.get_parameter("morph_kernel").value)

        self.canny1 = int(self.get_parameter("canny1").value)
        self.canny2 = int(self.get_parameter("canny2").value)
        self.hough_thr = int(self.get_parameter("hough_threshold").value)
        self.min_line_len = int(self.get_parameter("min_line_length").value)
        self.max_line_gap = int(self.get_parameter("max_line_gap").value)

        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.draw_hough_lines = bool(self.get_parameter("draw_hough_lines").value)

        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        self.get_logger().info(f"Load YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # class id
        self.target_id = None
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v) == str(self.target_class):
                    self.target_id = int(k); break
        elif isinstance(names, list):
            for i, v in enumerate(names):
                if str(v) == str(self.target_class):
                    self.target_id = int(i); break

        if self.target_id is None:
            self.get_logger().warn(f"target_class '{self.target_class}' not found -> no class filter")
        else:
            self.get_logger().info(f"Target class '{self.target_class}' id={self.target_id}")

        self._lut = gamma_lut(self.gamma)
        self._t_last = time.time()
        self._fps = 0.0

        self.sub = self.create_subscription(Image, self.image_topic, self.cb, qos_profile_sensor_data)
        self.pub_annot = self.create_publisher(Image, self.annotated_topic, qos_profile_sensor_data)
        self.pub_det = self.create_publisher(Float32MultiArray, self.detections_topic, 10)

        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Publish: {self.annotated_topic}, {self.detections_topic}")

    # --- Otsu on ROI (masked or not) ---
    def otsu_thr(self, gray_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> Optional[float]:
        if mask_u8 is None:
            thr, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return float(thr)
        vals = gray_u8[mask_u8 > 0]
        if vals.size < 80:
            return None
        vals = vals.reshape(-1, 1).astype(np.uint8)
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(thr)

    def make_bw(self, gray_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> Optional[np.ndarray]:
        thr = self.otsu_thr(gray_u8, mask_u8)
        if thr is None:
            return None
        bw = (gray_u8 > thr).astype(np.uint8) * 255
        if mask_u8 is not None:
            bw = cv2.bitwise_and(bw, bw, mask=mask_u8)

        k = max(3, int(self.morph_k))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        if mask_u8 is not None:
            bw = cv2.bitwise_and(bw, bw, mask=mask_u8)
        return bw

    def extract_lines(self, edges_u8: np.ndarray) -> List[Line]:
        lines = cv2.HoughLinesP(
            edges_u8, rho=1, theta=np.pi/180.0,
            threshold=self.hough_thr,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )
        out: List[Line] = []
        if lines is None:
            return out
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ln = segment_to_line(x1, y1, x2, y2)
            if ln is not None:
                out.append(ln)
        return out

    def fallback_minarearect(self, bw: np.ndarray) -> Optional[np.ndarray]:
        cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if float(cv2.contourArea(c)) < 100:
            return None
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        return order_points_clockwise(box)

    def yaw_from_quad(self, quad: np.ndarray) -> float:
        q = order_points_clockwise(quad)
        edges = q - np.roll(q, -1, axis=0)
        lens = np.linalg.norm(edges, axis=1)
        i = int(np.argmax(lens))
        p0 = q[i]
        p1 = q[(i + 1) % 4]
        yaw = math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))
        return map_to_pm90(yaw)

    # --- detection parsing: works with OBB or normal boxes ---
    def get_detections(self, res) -> List[Tuple[str, float, int, Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        returns list of:
        (mode, conf, cls, xyxy, poly4)
        mode: "obb" or "box"
        """
        dets = []

        # OBB
        if hasattr(res, "obb") and res.obb is not None and hasattr(res.obb, "xyxyxyxy") and res.obb.xyxyxyxy is not None:
            polys = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2).astype(np.float32)
            confs = res.obb.conf.cpu().numpy() if res.obb.conf is not None else np.ones((polys.shape[0],), dtype=np.float32)
            clss = res.obb.cls.cpu().numpy().astype(int) if res.obb.cls is not None else -np.ones((polys.shape[0],), dtype=np.int32)
            for p4, cf, cl in zip(polys, confs, clss):
                dets.append(("obb", float(cf), int(cl), None, p4))
            return dets

        # axis-aligned boxes
        if hasattr(res, "boxes") and res.boxes is not None and hasattr(res.boxes, "xyxy"):
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)
            confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
            clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else -np.ones((xyxy.shape[0],), dtype=np.int32)
            for b, cf, cl in zip(xyxy, confs, clss):
                dets.append(("box", float(cf), int(cl), b, None))
        return dets

    def process_roi(self, roi_bgr: np.ndarray, mask_u8: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, Tuple[float, float], float, List[Line]]]:
        # gray -> gamma -> blur
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray_g = cv2.LUT(gray, self._lut)
        gray_g = cv2.GaussianBlur(gray_g, (5, 5), 0).astype(np.uint8)

        bw = self.make_bw(gray_g, mask_u8)
        if bw is None:
            return None

        edges = cv2.Canny(bw, self.canny1, self.canny2, apertureSize=3)
        if mask_u8 is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask_u8)

        lines = self.extract_lines(edges)

        quad = best_parallelogram_from_lines(lines, roi_bgr.shape[1], roi_bgr.shape[0])
        if quad is None:
            quad = self.fallback_minarearect(bw)
            if quad is None:
                return None

        quad = order_points_clockwise(quad)
        cx = float(quad[:, 0].mean())
        cy = float(quad[:, 1].mean())
        yaw = self.yaw_from_quad(quad)

        return (quad, (cx, cy), yaw, lines)

    def cb(self, msg: Image):
        frame = rosimg_to_bgr(msg)
        vis = frame.copy()
        H, W = frame.shape[:2]

        # FPS
        t = time.time()
        dt = t - self._t_last
        if dt > 1e-6:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._t_last = t

        # YOLO
        res = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, max_det=self.max_det,
            verbose=False, device="cpu"
        )[0]

        dets = self.get_detections(res)
        if self.target_id is not None:
            dets = [d for d in dets if d[2] == self.target_id]
        dets = [d for d in dets if d[1] >= self.conf]
        dets.sort(key=lambda x: x[1], reverse=True)

        cv2.putText(vis, f"FPS: {self._fps:.1f}  n={len(dets)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        payload: List[float] = []

        for idx, (mode, conf, cls_id, xyxy, poly4) in enumerate(dets):
            if mode == "box":
                x0, y0, x1, y1 = xyxy
                x0 = int(max(0, math.floor(x0)))
                y0 = int(max(0, math.floor(y0)))
                x1 = int(min(W, math.ceil(x1)))
                y1 = int(min(H, math.ceil(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = frame[y0:y1, x0:x1].copy()
                mask = None
                # draw bbox
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

            else:  # OBB poly
                p4 = np.asarray(poly4, dtype=np.float32).reshape(4, 2)
                cv2.polylines(vis, [p4.astype(np.int32)], True, (0, 255, 255), 2)

                x, y, w, h = cv2.boundingRect(p4.astype(np.int32))
                x0 = max(0, x); y0 = max(0, y)
                x1 = min(W, x + w); y1 = min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = frame[y0:y1, x0:x1].copy()

                rh, rw = roi.shape[:2]
                mask = np.zeros((rh, rw), dtype=np.uint8)
                p4_roi = p4.copy()
                p4_roi[:, 0] -= x0
                p4_roi[:, 1] -= y0
                cv2.fillPoly(mask, [p4_roi.astype(np.int32)], 255)

            out = self.process_roi(roi, mask)
            if out is None:
                continue
            quad_roi, (cx_roi, cy_roi), yaw, lines = out

            # optional draw hough lines
            if self.draw_hough_lines:
                # draw on ROI then paste (cheap debug)
                roi_dbg = roi.copy()
                # need edges to draw? we draw segments from original Hough? (we only kept Line)
                # skip detailed segments; just show quad
                cv2.polylines(roi_dbg, [quad_roi.astype(np.int32)], True, (0, 255, 0), 3)
                vis[y0:y1, x0:x1] = roi_dbg

            # convert quad to image coords
            quad_img = quad_roi.copy()
            quad_img[:, 0] += x0
            quad_img[:, 1] += y0

            cx = cx_roi + x0
            cy = cy_roi + y0

            cv2.polylines(vis, [quad_img.astype(np.int32)], True, (0, 255, 0), 3)
            icx, icy = int(round(cx)), int(round(cy))
            cv2.circle(vis, (icx, icy), 6, (0, 0, 255), -1)
            draw_arrow(vis, (icx, icy), yaw, length=80)

            cv2.putText(vis, f"{self.target_class} {conf:.2f} yaw:{yaw:.1f}",
                        (icx + 8, icy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # publish: [id, cx, cy, yaw_deg, conf]
            payload += [float(idx), float(cx), float(cy), float(yaw), float(conf)]

        arr = Float32MultiArray()
        arr.data = payload
        self.pub_det.publish(arr)
        self.pub_annot.publish(bgr_to_rosimg(vis, msg.header))


def main():
    rclpy.init()
    node = YoloHoughParallelogram()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()