# live_cam.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av  # provided by 'av' package
import numpy as np
import cv2
import time

st.set_page_config(page_title="Live Camera", layout="wide")
st.title("🔴 Live Webcam (streamlit-webrtc)")

class EchoVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_ts = time.time()
        self.fps = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert VideoFrame -> numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # ---- demo: draw FPS and a red border so you can see it’s live ----
        now = time.time()
        dt = now - self.last_ts
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_ts = now

        h, w = img.shape[:2]
        cv2.rectangle(img, (10, 10), (w-10, h-10), (0, 0, 255), 2)
        cv2.putText(img, f"FPS: {self.fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        # Convert back to VideoFrame to stream out
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EchoVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,   # don't block the UI thread
)
