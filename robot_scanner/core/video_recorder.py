import os
import threading

import cv2


class KinectFrameRecorder:
    """Saves Kinect color captures as numbered PNGs into output_dir."""

    def __init__(self, kinect, output_dir):
        self.kinect = kinect
        self.output_dir = output_dir
        self._thread = None
        self._stop = threading.Event()
        self._frames_written = 0

    def _loop(self):
        os.makedirs(self.output_dir, exist_ok=True)
        while not self._stop.is_set():
            capture = self.kinect.sensor.get_capture()
            if capture.color is None:
                continue
            bgr = capture.color[:, :, :3]
            cv2.imwrite(os.path.join(self.output_dir, f"{self._frames_written:06d}.png"), bgr)
            self._frames_written += 1

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[FrameRecorder] Saving frames to {self.output_dir}")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        print(f"[FrameRecorder] Wrote {self._frames_written} frames to {self.output_dir}")
