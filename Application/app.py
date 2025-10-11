# app.py
import json, io, os, time
from pathlib import Path
from typing import List
from uuid import uuid4

import av
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Optional deps (AWS + charting)
try:
    import boto3 as _boto3
    from boto3.dynamodb.conditions import Attr as _Attr
except Exception:
    _boto3 = None
    _Attr = None

try:
    import pandas as pd
except Exception:
    pd = None

# ────────────────────────────────────────────────────────────────────────────────
# Page & helpers
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Detection (ONNX)", page_icon="🎥", layout="wide")
st.title("🎥 Emotion Detection (ONNX) — Image & Live Webcam")

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

def resolve(p: str | Path) -> Path:
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        return p
    p1 = (APP_DIR / p).resolve()
    if p1.exists():
        return p1
    p2 = (MODELS_DIR / p).resolve()
    if p2.exists():
        return p2
    return p.resolve()

# ────────────────────────────────────────────────────────────────────────────────
# Paths UI
# ────────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    onnx_in = st.text_input("ONNX model path", str(MODELS_DIR / "model.onnx"))
with c2:
    names_in = st.text_input("Class names JSON", str(MODELS_DIR / "class_names.json"))

onnx_path = resolve(onnx_in)
names_path = resolve(names_in)

st.caption(f"Model: `{onnx_path}`")
st.caption(f"Classes: `{names_path}`")

if not onnx_path.exists() or not names_path.exists():
    st.error("Model or class_names file not found at the resolved paths above.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Preprocessing controls
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("Preprocessing", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        swap_bgr_rgb = st.selectbox("Color", ["BGR→RGB", "Keep BGR"], index=0) == "BGR→RGB"
        input_side = st.number_input("Input size (square)", 64, 512, 224, 8)
        scale_01 = st.checkbox("Scale to [0,1]", value=True)
    with colB:
        letterbox = st.checkbox("Letterbox (keep aspect)", value=True)
        center_crop = st.checkbox("Center crop to square", value=False)  # face-crop is always on
    with colC:
        mean_txt = st.text_input("Mean (comma)", "0.485,0.456,0.406")
        std_txt  = st.text_input("Std (comma)",  "0.229,0.224,0.225")

with st.expander("Smoothing / display", expanded=True):
    show_topk = st.slider("Show top-k (sidebar text)", 1, 5, 3)
    smooth = st.checkbox("Temporal smoothing (webcam)", value=True)
    smooth_win = st.slider("Smoothing window", 1, 20, 5)
    conf_fmt = st.text_input("Confidence format", "{:.2f}")

# NEW: overlay style controls (font sizes + thickness)
with st.expander("Overlay style", expanded=True):
    fs_top1 = st.slider("Font scale — #1 (top prediction)", 0.5, 4.0, 1.6, 0.1)
    fs_top2 = st.slider("Font scale — #2",                 0.5, 4.0, 1.3, 0.1)
    fs_top3 = st.slider("Font scale — #3",                 0.5, 4.0, 1.1, 0.1)
    th_top  = st.slider("Font thickness (all)",            1,   6,   2)

with st.expander("Advanced (ONNX session)", expanded=False):
    single_thread = st.checkbox("Single-threaded session (helps some Windows setups)", value=False)
    providers = st.multiselect("Execution providers (ordered)", ["CPUExecutionProvider"], default=["CPUExecutionProvider"])

# ────────────────────────────────────────────────────────────────────────────────
# ✅ AWS INTEGRATION
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("AWS Integration (optional)", expanded=True):
    st.write("Uses your local AWS credentials (from AWS CLI or environment).")
    col1, col2 = st.columns(2)
    with col1:
        enable_s3 = st.checkbox("Save annotated snapshots to S3 (+ JSON logs)", value=True)
        s3_bucket = st.text_input(
            "S3 bucket",
            os.getenv("APP_S3_BUCKET", "emotion-app-729798775712-ap-southeast-2"),
        )
        s3_every_sec = st.number_input("Snapshot/log interval (seconds)", 1, 600, 5, 1)
    with col2:
        enable_ddb = st.checkbox("Also log predictions to DynamoDB", value=False)
        ddb_table = st.text_input(
            "DynamoDB table",
            os.getenv("APP_DDB_TABLE", "emotion_logs"),
        )
        ddb_every_sec = st.number_input("DDB min log interval (seconds)", 0, 60, 1, 1)

# Best-effort boto3 init (lazy if disabled)
boto3 = None
s3 = None
ddb = None
Attr = _Attr
aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-2"

if enable_s3 or enable_ddb:
    try:
        boto3 = _boto3
        if boto3:
            if enable_s3:
                s3 = boto3.client("s3", region_name=aws_region)
            if enable_ddb:
                ddb = boto3.resource("dynamodb", region_name=aws_region).Table(ddb_table)
    except Exception as _e:
        st.warning(f"AWS SDK (boto3) not available or failed to init: {_e}")

# ────────────────────────────────────────────────────────────────────────────────
# Load classes & model
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classes(p: Path) -> List[str]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "classes" in data:
        return list(map(str, data["classes"]))
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda x: int(x))
        return [str(data[k]) for k in keys]
    return list(map(str, data))

try:
    class_names = load_classes(names_path)
except Exception as e:
    st.error(f"Failed to parse class_names.json: {e}")
    st.stop()

@st.cache_resource
def load_sess(path: Path, single_threaded: bool, providers: List[str]):
    if single_threaded:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    else:
        sess = ort.InferenceSession(str(path), providers=providers)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    shp = list(inp.shape)
    H = shp[-2] if isinstance(shp[-2], int) else input_side
    W = shp[-1] if isinstance(shp[-1], int) else input_side
    return sess, inp.name, out.name, (H, W)

try:
    sess, input_name, output_name, (H_model, W_model) = load_sess(onnx_path, single_thread, providers)
except Exception as e:
    st.error(f"ONNX load error: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Haar: always load (cropping is always applied)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_cascade()

# ────────────────────────────────────────────────────────────────────────────────
# Preprocess / postprocess
# ────────────────────────────────────────────────────────────────────────────────
def parse3(txt: str) -> np.ndarray:
    try:
        vals = [float(x.strip()) for x in txt.split(",")]
        assert len(vals) == 3
        return np.array(vals, dtype=np.float32).reshape(1,1,3)
    except Exception:
        return np.array([[[0.485,0.456,0.406]]], dtype=np.float32)

mean = parse3(mean_txt)
std  = parse3(std_txt)

def do_letterbox(bgr: np.ndarray, side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = side / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - nh) // 2
    x0 = (side - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return canvas

# Face helpers (largest face + draw box)
def detect_largest_face(bgr: np.ndarray):
    if face_cascade is None:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # (x,y,w,h)

def draw_face_box(img: np.ndarray, bbox, pad_ratio: float = 0.15):
    if bbox is None:
        return
    x, y, w, h = map(int, bbox)
    pad = int(pad_ratio * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img.shape[1] - 1, x + w + pad); y1 = min(img.shape[0] - 1, y + h + pad)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

def face_crop_region(bgr: np.ndarray) -> np.ndarray:
    """Always try to crop to the largest detected face; fall back to original if none."""
    bbox = detect_largest_face(bgr)
    if bbox is None:
        return bgr
    x, y, w, h = map(int, bbox)
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad); y1 = min(bgr.shape[0], y + h + pad)
    return bgr[y0:y1, x0:x1]

def preprocess(bgr: np.ndarray) -> np.ndarray:
    # Always crop to face if detected
    img = face_crop_region(bgr)

    # Optional square shaping
    if center_crop:
        h, w = img.shape[:2]
        m = min(h, w)
        y0 = (h - m)//2; x0 = (w - m)//2
        img = img[y0:y0+m, x0:x0+m]
        img = cv2.resize(img, (input_side, input_side), interpolation=cv2.INTER_AREA)
    else:
        img = do_letterbox(img, input_side) if letterbox else cv2.resize(img, (input_side, input_side), interpolation=cv2.INTER_AREA)

    # Color + normalize
    if swap_bgr_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32)
    if scale_01:
        x /= 255.0
    x = (x - mean) / std

    # Match model declared HxW if needed
    if (H_model, W_model) != (input_side, input_side):
        x = cv2.resize(x, (W_model, H_model), interpolation=cv2.INTER_LINEAR)

    # NCHW
    x = np.transpose(x, (2,0,1))[None, ...].astype(np.float32)  # [1,3,H,W]
    return x

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=axis, keepdims=True)

def topk(probs: np.ndarray, k=3):
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

# ────────────────────────────────────────────────────────────────────────────────
# Single image
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("📷 Single image")
img_up = st.file_uploader("Upload a face image (jpg/png)", type=["jpg","jpeg","png","bmp"])
if img_up is not None:
    arr = np.frombuffer(img_up.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Failed to decode image.")
    else:
        # Show original
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input (RGB)")

        # Show annotated with face box and top-3 text using the same UI scales
        bgr_annot = bgr.copy()
        draw_face_box(bgr_annot, detect_largest_face(bgr))
        try:
            x = preprocess(bgr)
            logits = sess.run([output_name], {input_name: x})[0]  # [1,C]
            probs = softmax(logits, axis=1)[0]
            best = int(np.argmax(probs))
            st.success(f"Prediction: **{class_names[best]}**  ({conf_fmt.format(float(probs[best]))})")

            # Draw top-3 on the annotated image
            top = topk(probs, k=min(3, len(class_names)))
            scales = [fs_top1, fs_top2, fs_top3]
            y = 40
            def draw_text(img, text, org=(18, 42), scale=1.2, thickness=2):
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness,   cv2.LINE_AA)
            for rank, (i, p) in enumerate(top):
                tag = f"#{rank+1} {class_names[i]}: {conf_fmt.format(p)}"
                draw_text(bgr_annot, tag, (20, y), scale=scales[rank], thickness=th_top)
                y += int(38 * scales[rank])

            st.image(cv2.cvtColor(bgr_annot, cv2.COLOR_BGR2RGB), caption="Detected face + Top-3")

            # Also show numeric top-k list
            st.write("Top-k:", {class_names[i]: conf_fmt.format(p) for i, p in topk(probs, k=min(show_topk, len(class_names)))})
        except Exception as e:
            st.error(f"Inference error: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Webcam (WebRTC)
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("🎛 Webcam (WebRTC)")

# Sanity check output size vs class count (informative)
try:
    dummy = np.zeros((1,3,H_model,W_model), dtype=np.float32)
    test_logits = sess.run([output_name], {input_name: dummy})[0]
    outC = int(test_logits.shape[-1])
    if outC != len(class_names):
        st.warning(f"Model outputs {outC} classes but class_names has {len(class_names)}. Check class_names.json order/length.")
except Exception as _e:
    st.info(f"Could not probe output shape: {_e}")

def draw_text(img, text, org=(18, 42), scale=1.2, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness,   cv2.LINE_AA)

class _Smoother:
    def __init__(self, win: int):
        self.win = max(1, int(win)); self.buf: List[np.ndarray] = []
    def push(self, p: np.ndarray) -> np.ndarray:
        self.buf.append(p)
        if len(self.buf) > self.win:
            self.buf.pop(0)
        return np.mean(self.buf, axis=0)

class ONNXProcessor(VideoProcessorBase):
    def __init__(self):
        self.smoother = _Smoother(smooth_win) if smooth else None
        self._last_snap = 0.0
        self._last_log  = 0.0

    def _s3_snapshot_and_log(self, frame_bgr: np.ndarray, label: str, conf: float):
        if not (s3 and enable_s3 and s3_bucket):
            return
        now = time.monotonic()
        if now - self._last_snap < float(s3_every_sec):
            return
        self._last_snap = now

        # 1) Annotated JPG under frames/<label>/YYYY/MM/DD/HH/...
        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if ok:
            ts = int(time.time())
            dt = time.strftime("%Y/%m/%d/%H", time.gmtime(ts))
            img_key = f"frames/{label}/{dt}/{ts}_{uuid4().hex[:8]}.jpg"
            try:
                s3.put_object(
                    Bucket=s3_bucket,
                    Key=img_key,
                    Body=io.BytesIO(jpg.tobytes()).getvalue(),
                    ContentType="image/jpeg",
                )
            except Exception:
                pass

        # 2) Tiny JSON log under logs/YYYY/MM/DD/HH/...
        try:
            ts = int(time.time())
            dt = time.strftime("%Y/%m/%d/%H", time.gmtime(ts))
            log_key = f"logs/{dt}/{ts}_{uuid4().hex[:8]}.json"
            payload = {"ts": ts, "label": label, "conf": float(conf)}
            s3.put_object(
                Bucket=s3_bucket,
                Key=log_key,
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception:
            pass

    def _ddb_log(self, label: str, conf: float):
        if not (ddb and enable_ddb and ddb_table):
            return
        now = time.monotonic()
        if now - self._last_log < float(ddb_every_sec):
            return
        self._last_log = now
        try:
            ddb.put_item(Item={
                "id": str(uuid4()),
                "ts": int(time.time()),
                "label": label,
                "conf": float(conf),
            })
        except Exception:
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            bgr = frame.to_ndarray(format="bgr24")

            # Always crop to face for inference
            x = preprocess(bgr)
            logits = sess.run([output_name], {input_name: x})[0]
            probs = softmax(logits, axis=1)[0]
            if self.smoother:
                probs = self.smoother.push(probs)
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = class_names[idx]

            out = bgr.copy()

            # Draw face box on the live frame (for display)
            draw_face_box(out, detect_largest_face(bgr))

            # Overlay top-3 predictions with adjustable font sizes
            top = topk(probs, k=min(3, len(class_names)))
            scales = [fs_top1, fs_top2, fs_top3]
            y = 50
            for rank, (i, p) in enumerate(top):
                tag = f"#{rank+1} {class_names[i]}: {conf_fmt.format(p)}"
                draw_text(out, tag, (20, y), scale=scales[rank], thickness=th_top)
                y += int(38 * scales[rank])  # spacing matches size

            # S3/DDB logging (annotated frame)
            self._s3_snapshot_and_log(out, label, conf)
            self._ddb_log(label, conf)

            return av.VideoFrame.from_ndarray(out, format="bgr24")
        except Exception as e:
            out = frame.to_ndarray(format="bgr24").copy()
            draw_text(out, f"ERR: {e.__class__.__name__}", (20, 50))
            return av.VideoFrame.from_ndarray(out, format="bgr24")

webrtc_streamer(
    key="onnx-webcam",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ONNXProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# 📈 Emotion frequency (DynamoDB or S3)
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Emotion frequency")

def fetch_counts_ddb(table, since_minutes: int = 60) -> dict:
    if not table or not boto3:
        return {}
    now = int(time.time())
    cutoff = now - since_minutes * 60
    items = []
    kwargs = {
        "ProjectionExpression": "#lb, #ts",
        "ExpressionAttributeNames": {"#lb": "label", "#ts": "ts"},
    }
    use_filter = _Attr is not None
    if use_filter:
        kwargs["FilterExpression"] = _Attr("ts").gte(cutoff)
    while True:
        resp = table.scan(**kwargs)
        batch = resp.get("Items", [])
        if not use_filter:
            batch = [it for it in batch if int(it.get("ts", 0)) >= cutoff]
        items.extend(batch)
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    counts = {}
    for it in items:
        lb = (it.get("label") or "unknown")
        counts[lb] = counts.get(lb, 0) + 1
    return counts

def iter_s3_objects(client, bucket: str, prefix: str):
    """Generator over all objects under a prefix."""
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp.get("NextContinuationToken")

def fetch_counts_s3(client, bucket: str, since_minutes: int = 60) -> dict:
    """
    Prefer JSON logs under logs/YYYY/MM/DD/HH/*.json.
    If none exist, fall back to counting frames/<label>/... keys by LastModified.
    """
    if not client or not bucket:
        return {}
    cutoff = time.time() - since_minutes * 60

    # 1) Try logs:
    counts = {}
    try:
        prefix = "logs/"
        for obj in iter_s3_objects(client, bucket, prefix):
            if obj["LastModified"].timestamp() < cutoff:
                continue
            key = obj["Key"]
            try:
                body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
                data = json.loads(body.decode("utf-8"))
                label = str(data.get("label", "unknown"))
                counts[label] = counts.get(label, 0) + 1
            except Exception:
                continue
        if counts:
            return counts
    except Exception:
        pass

    # 2) Fallback: infer from frames/<label>/... keys
    try:
        prefix = "frames/"
        for obj in iter_s3_objects(client, bucket, prefix):
            if obj["LastModified"].timestamp() < cutoff:
                continue
            key = obj["Key"]  # frames/<label>/YYYY/...
            parts = key.split("/")
            if len(parts) >= 3:
                label = parts[1] or "unknown"
                counts[label] = counts.get(label, 0) + 1
    except Exception:
        pass
    return counts

colA, colB, colC = st.columns([1,1,1])
with colA:
    window_min = st.number_input("Lookback window (minutes)", 1, 1440, 60, 1)
with colB:
    if st.button("Refresh"):
        pass
with colC:
    st.caption(f"Region: {aws_region}")

# Prefer DDB if enabled and available; else use S3
counts = {}
if ddb and enable_ddb:
    counts = fetch_counts_ddb(ddb, since_minutes=int(window_min))
elif s3 and enable_s3:
    counts = fetch_counts_s3(s3, s3_bucket, since_minutes=int(window_min))

if not counts:
    st.info("No records found in the selected window.")
else:
    if pd:
        df = pd.DataFrame(
            [{"emotion": k, "count": v} for k, v in sorted(counts.items())]
        ).set_index("emotion")
        st.bar_chart(df)
        with st.expander("Raw counts"):
            st.json(counts)
    else:
        st.write(counts)
        st.caption("Install pandas for a nicer chart: `pip install pandas`")
