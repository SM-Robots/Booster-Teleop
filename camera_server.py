#!/usr/bin/env python3
"""Teleop bridge server for Booster K1 robot.

Endpoints:
  GET  /stream        — MJPEG camera stream
  GET  /health        — server health
  GET  /robot/state   — JSON: mode, IMU, odometry
  POST /robot/command — JSON: move, head, mode, get_up, lie_down, stop

SDK runs in a child process to avoid DDS conflicts with rclpy.
State shared via a raw shared memory buffer (no Manager).
"""

import json
import multiprocessing
import multiprocessing.shared_memory
import signal
import socket
import struct
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
JPEG_QUALITY = 70

# --- Shared state -----------------------------------------------------------
_frame_lock = threading.Lock()
_frame_cond = threading.Condition(_frame_lock)
_latest_jpeg = None
_frame_count = 0
_last_frame_time = 0.0

# SDK state: 10 floats (rpy*3 + gyro*3 + acc*3 + timestamp) + odom 3 floats
# = 13 doubles = 104 bytes, plus 1 byte for mode
STATE_FMT = "=13db"  # 13 doubles + 1 byte
STATE_SIZE = struct.calcsize(STATE_FMT)

_state_shm = None  # shared memory
_cmd_queue = None
_resp_queue = None


def nv12_to_jpeg(data, width, height):
    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def pack_state(rpy, gyro, acc, odom_x, odom_y, odom_theta, ts, mode_byte):
    return struct.pack(STATE_FMT,
                       rpy[0], rpy[1], rpy[2],
                       gyro[0], gyro[1], gyro[2],
                       acc[0], acc[1], acc[2],
                       odom_x, odom_y, odom_theta, ts, mode_byte)


def unpack_state(buf):
    vals = struct.unpack(STATE_FMT, buf)
    return {
        "imu": {
            "rpy": [round(vals[0], 4), round(vals[1], 4), round(vals[2], 4)],
            "gyro": [round(vals[3], 4), round(vals[4], 4), round(vals[5], 4)],
            "acc": [round(vals[6], 4), round(vals[7], 4), round(vals[8], 4)],
        },
        "odometry": {
            "x": round(vals[9], 4), "y": round(vals[10], 4),
            "theta": round(vals[11], 4),
        },
        "timestamp": vals[12],
        "mode_byte": vals[13],
    }


# --- SDK child process -------------------------------------------------------
def _sdk_worker(cmd_q, resp_q, shm_name):
    """Separate process: owns SDK DDS."""
    from booster_robotics_sdk_python import (
        ChannelFactory, B1LocoClient, B1LowStateSubscriber,
        B1OdometerStateSubscriber, RobotMode, GetModeResponse,
    )
    MODE_MAP = {
        "damping": RobotMode.kDamping, "prepare": RobotMode.kPrepare,
        "walking": RobotMode.kWalking, "custom": RobotMode.kCustom,
    }
    MODE_NAMES = {v: k for k, v in MODE_MAP.items()}

    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    imu_data = [0.0] * 9  # rpy*3 + gyro*3 + acc*3
    odom_data = [0.0, 0.0, 0.0]
    ts = [0.0]

    ChannelFactory.Instance().Init(0)
    client = B1LocoClient()
    client.Init()

    def on_low_state(msg):
        imu = msg.imu_state
        imu_data[:3] = imu.rpy
        imu_data[3:6] = imu.gyro
        imu_data[6:9] = imu.acc
        ts[0] = time.time()
        shm.buf[:STATE_SIZE] = pack_state(
            imu.rpy, imu.gyro, imu.acc,
            odom_data[0], odom_data[1], odom_data[2], ts[0], 0)

    def on_odometer(msg):
        odom_data[:] = [msg.x, msg.y, msg.theta]

    low_sub = B1LowStateSubscriber(on_low_state)
    low_sub.InitChannel()
    odom_sub = B1OdometerStateSubscriber(on_odometer)
    odom_sub.InitChannel()

    while True:
        try:
            data = cmd_q.get()
        except Exception:
            break
        cmd = data.get("command")
        res = 0
        try:
            if cmd == "move":
                res = client.Move(float(data.get("vx", 0)),
                                  float(data.get("vy", 0)),
                                  float(data.get("vyaw", 0)))
            elif cmd == "head":
                res = client.RotateHead(float(data.get("pitch", 0)),
                                        float(data.get("yaw", 0)))
            elif cmd == "mode":
                mode = MODE_MAP.get(data.get("mode"))
                res = client.ChangeMode(mode) if mode else -1
            elif cmd == "get_up":
                res = client.GetUp()
            elif cmd == "lie_down":
                res = client.LieDown()
            elif cmd == "stop":
                res = client.Move(0, 0, 0)
            elif cmd == "get_mode":
                gm = GetModeResponse()
                res = client.GetMode(gm)
                resp_q.put({"command": cmd, "result": res,
                            "mode": MODE_NAMES.get(gm.mode, str(gm.mode))})
                continue
            else:
                resp_q.put({"error": f"unknown command: {cmd}"})
                continue
        except Exception as e:
            resp_q.put({"error": str(e)})
            continue
        resp_q.put({"command": cmd, "result": res})


# --- ROS2 node (camera only) -------------------------------------------------
class CameraNode(Node):
    def __init__(self):
        super().__init__("teleop_bridge")
        self.create_subscription(
            Image, "/StereoNetNode/rectified_image", self._on_image, 1)

    def _on_image(self, msg):
        global _latest_jpeg, _frame_count, _last_frame_time
        try:
            jpeg = nv12_to_jpeg(msg.data, msg.width, msg.height)
        except Exception:
            return
        with _frame_cond:
            _latest_jpeg = jpeg
            _frame_count += 1
            _last_frame_time = time.time()
            _frame_cond.notify_all()


# --- HTTP handler -------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        routes = {"/stream": self._stream, "/health": self._health,
                  "/robot/state": self._get_state}
        h = routes.get(self.path)
        if h:
            h()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/robot/command":
            self._post_command()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        last_sent = 0
        try:
            while True:
                with _frame_cond:
                    _frame_cond.wait(timeout=2.0)
                    jpeg = _latest_jpeg
                    fc = _frame_count
                if jpeg is None or fc == last_sent:
                    continue
                last_sent = fc
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode()
                    + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _health(self):
        age = time.time() - _last_frame_time if _last_frame_time else -1
        self._json(200, {
            "status": "ok" if 0 < age < 5 else "no_frames",
            "frame_count": _frame_count,
            "last_frame_age_s": round(age, 2),
        })

    def _get_state(self):
        # Read IMU/odom from shared memory
        state = unpack_state(bytes(_state_shm.buf[:STATE_SIZE]))
        # Get mode from SDK process
        _cmd_queue.put({"command": "get_mode"})
        try:
            resp = _resp_queue.get(timeout=2.0)
            state["mode"] = resp.get("mode", "unknown")
        except Exception:
            state["mode"] = "unknown"
        del state["mode_byte"]
        self._json(200, state)

    def _post_command(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid json"})
            return
        _cmd_queue.put(data)
        try:
            resp = _resp_queue.get(timeout=3.0)
        except Exception:
            resp = {"error": "timeout"}
        code = 200 if "error" not in resp else 400
        self._json(code, resp)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, fmt, *args):
        pass


# --- Main --------------------------------------------------------------------
def main():
    global _cmd_queue, _resp_queue, _state_shm

    # Shared memory for state (no Manager process needed)
    _state_shm = multiprocessing.shared_memory.SharedMemory(
        create=True, size=STATE_SIZE)
    _state_shm.buf[:STATE_SIZE] = b'\x00' * STATE_SIZE

    _cmd_queue = multiprocessing.Queue()
    _resp_queue = multiprocessing.Queue()

    # SDK in child process
    sdk_proc = multiprocessing.Process(
        target=_sdk_worker,
        args=(_cmd_queue, _resp_queue, _state_shm.name),
        daemon=True)
    sdk_proc.start()

    # ROS2 in main process (camera only)
    rclpy.init()
    node = CameraNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    # HTTP server
    server = HTTPServer((HOST, PORT), Handler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def shutdown(sig, _):
        server.shutdown()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"Teleop bridge listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()
    server.server_close()
    sdk_proc.kill()
    _state_shm.close()
    _state_shm.unlink()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
