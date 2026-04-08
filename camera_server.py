#!/usr/bin/env python3
"""MJPEG streaming server for Booster K1 robot camera.

Subscribes to /StereoNetNode/rectified_image (NV12 544x448),
converts to JPEG, and serves as MJPEG at :8080/stream.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

HOST = "0.0.0.0"
PORT = 8080
TOPIC = "/StereoNetNode/rectified_image"
JPEG_QUALITY = 70

# Shared state: latest JPEG frame + condition for signaling new frames
_frame_lock = threading.Lock()
_frame_cond = threading.Condition(_frame_lock)
_latest_jpeg: bytes | None = None
_frame_count = 0
_last_frame_time = 0.0


def nv12_to_jpeg(data: bytes, width: int, height: int) -> bytes:
    """Convert NV12 YUV frame to JPEG bytes."""
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes()


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_mjpeg_server")
        self.create_subscription(Image, TOPIC, self._on_image, 1)
        self.get_logger().info(f"Subscribed to {TOPIC}")

    def _on_image(self, msg: Image):
        global _latest_jpeg, _frame_count, _last_frame_time
        try:
            jpeg = nv12_to_jpeg(msg.data, msg.width, msg.height)
        except Exception as e:
            self.get_logger().warn(f"Frame convert error: {e}")
            return
        with _frame_cond:
            _latest_jpeg = jpeg
            _frame_count += 1
            _last_frame_time = time.time()
            _frame_cond.notify_all()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream":
            self._stream()
        elif self.path == "/health":
            self._health()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with _frame_cond:
                    _frame_cond.wait(timeout=2.0)
                    jpeg = _latest_jpeg
                if jpeg is None:
                    continue
                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _health(self):
        age = time.time() - _last_frame_time if _last_frame_time else -1
        body = json.dumps({
            "status": "ok" if 0 < age < 5 else "no_frames",
            "frame_count": _frame_count,
            "last_frame_age_s": round(age, 2),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silence per-request logs


def ros_spin(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def main():
    rclpy.init()
    node = CameraNode()

    t = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    t.start()

    server = HTTPServer((HOST, PORT), Handler)
    print(f"MJPEG server listening on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
