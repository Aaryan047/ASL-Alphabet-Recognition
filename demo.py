"""
demo.py — ASL Live Webcam Demo (Browser-based)
- NOTHING  → does nothing (null, never written)
- SPACE    → commits current word to sentence, adds space
- DELETE   → removes last character from current word
- Letters  → build current word character by character
- UI       → dark/light toggle, current word panel, add-word button, full sentence below
"""

import os, json, time, threading
import cv2, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, Response, render_template_string, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_DIR             = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(_DIR, "backupmodel.pth")
LABELS_PATH      = os.path.join(_DIR, "backuplabels.json")

CAMERA_INDEX     = 1
NUM_CLASSES      = 29
IMG_SIZE         = 128

CONFIDENCE_THRES = 0.85
HOLD_SECONDS     = 0.75
STREAK_NEEDED    = 6

CONTROL_LABELS   = {"NOTHING", "nothing", "Nothing"}
# Labels that are actions, never appended as characters — matched case-insensitively
ACTION_LABELS    = {"space", "delete"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    net = models.mobilenet_v2(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(net.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.to(DEVICE).eval()
    return net, labels

net, LABELS = load_model()
print(f"Model ready on {DEVICE}")

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = PREPROCESS(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(net(t), 1)[0]
    idx = int(p.argmax())
    return LABELS[idx], float(p[idx])


# ── Hold state ────────────────────────────────────────────────────────────────
_hold = {"label": None, "streak": 0, "timer": None, "done": False}

def process_prediction(label, conf):
    now = time.time()

    # Hard gate: NOTHING or low confidence → full reset
    if label in CONTROL_LABELS or conf < CONFIDENCE_THRES:
        _hold["label"]  = None
        _hold["streak"] = 0
        _hold["timer"]  = None
        _hold["done"]   = False
        return 0.0

    # Different label → restart streak
    if label != _hold["label"]:
        _hold["label"]  = label
        _hold["streak"] = 1
        _hold["timer"]  = None
        _hold["done"]   = False
        return 0.0

    # Same label — grow streak
    _hold["streak"] += 1

    # Start timer once streak threshold reached
    if _hold["streak"] == STREAK_NEEDED:
        _hold["timer"] = now

    if _hold["timer"] is None or _hold["done"]:
        return 0.0

    elapsed  = now - _hold["timer"]
    progress = min(elapsed / HOLD_SECONDS, 1.0)

    if elapsed >= HOLD_SECONDS:
        with state_lock:
            label_lower = label.lower()
            if label_lower == "space":
                # Insert a literal whitespace character into the sentence
                state["sentence"] = state["sentence"] + " " #TS is not working
            elif label_lower == "delete":
                # Remove last char from current word
                state["current_word"] = state["current_word"][:-1]
            elif label not in CONTROL_LABELS and label_lower not in ACTION_LABELS:
                state["current_word"] += label

        _hold["done"]  = True
        _hold["timer"] = None
        return 0.0

    return progress


# ── Shared UI state ───────────────────────────────────────────────────────────
state_lock = threading.Lock()
state = {
    "label":        "",
    "confidence":   0.0,
    "current_word": "",   # word being built letter by letter
    "sentence":     "",   # committed sentence
    "progress":     0.0,
    "hint":         "",   # user guidance hint
}

def compute_hint(roi_bgr, label, conf):
    """Return a guidance string based on frame quality and prediction."""
    # Low-light check: mean brightness of the ROI
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 40:
        return "Low Light — move to a brighter area"

    # Too-close check: if >60% of the ROI pixels are near-skin-tone saturation
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    skin_mask = (h >= 0) & (h <= 25) & (s >= 40) & (v >= 80)
    skin_ratio = float(skin_mask.mean())
    if skin_ratio > 0.70:
        return "Too close — bring your hand further back"

    # Too-far / unclear: decent brightness, but confidence is low and a real sign attempted
    if label not in CONTROL_LABELS and label.lower() not in ACTION_LABELS and conf < CONFIDENCE_THRES:
        if conf < 0.45:
            return "Bring your hand closer"
        else:
            return "Hold still — sign not recognised"

    return ""


# ── Camera loop ───────────────────────────────────────────────────────────────
frame_lock   = threading.Lock()
latest_frame = None

def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display   = cv2.flip(raw, 1)
        h, w      = display.shape[:2]
        sz        = min(h, w) // 2
        cx, cy    = w // 2, h // 2
        x1, y1    = cx - sz // 2, cy - sz // 2
        x2, y2    = cx + sz // 2, cy + sz // 2
        roi_disp  = display[y1:y2, x1:x2]
        roi_model = cv2.flip(roi_disp, 1)

        label, conf = predict(roi_model)
        progress    = process_prediction(label, conf)
        hint        = compute_hint(roi_disp, label, conf)

        with state_lock:
            state["label"]      = label
            state["confidence"] = conf
            state["progress"]   = progress
            state["hint"]       = hint

        active = (label not in CONTROL_LABELS) and (label.lower() not in ACTION_LABELS) and (conf >= CONFIDENCE_THRES)
        color  = (0, 220, 80) if active else (70, 70, 70)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        if progress > 0:
            filled = int(sz * progress)
            cv2.rectangle(display, (x1, y1 - 8), (x1 + filled, y1 - 2), (0, 190, 255), -1)

        with frame_lock:
            latest_frame = display.copy()

    cap.release()

threading.Thread(target=camera_loop, daemon=True).start()


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASL Live Demo</title>
<style>
/* ── Theme tokens ── */
:root[data-theme="dark"] {
  --bg:        #0d0d0f;
  --bg2:       #16161a;
  --bg3:       #0a0a0c;
  --border:    #242428;
  --text:      #f0f0f0;
  --text2:     #888;
  --text3:     #444;
  --card-sh:   0 2px 12px rgba(0,0,0,.4);
}
:root[data-theme="light"] {
  --bg:        #f4f4f6;
  --bg2:       #ffffff;
  --bg3:       #ebebee;
  --border:    #dddde0;
  --text:      #111114;
  --text2:     #666;
  --text3:     #aaa;
  --card-sh:   0 2px 12px rgba(0,0,0,.08);
}

* { box-sizing: border-box; margin: 0; padding: 0; transition: background .2s, color .2s, border-color .2s; }

body {
  background: var(--bg); color: var(--text);
  font-family: 'Segoe UI', sans-serif;
  min-height: 100vh; padding: 24px 20px; gap: 0;
}

/* ── Top bar ── */
.topbar {
  display: flex; justify-content: space-between; align-items: center;
  max-width: 1100px; margin: 0 auto 24px;
}
.topbar h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text2); }

/* Dark/light toggle */
.toggle-wrap { display: flex; align-items: center; gap: 8px; }
.toggle-label { font-size: .75rem; color: var(--text3); }
.toggle {
  position: relative; width: 44px; height: 24px;
  background: var(--border); border-radius: 12px; cursor: pointer;
  transition: background .2s;
}
.toggle.on { background: #3b82f6; }
.toggle::after {
  content: ''; position: absolute; top: 3px; left: 3px;
  width: 18px; height: 18px; border-radius: 50%;
  background: #fff; transition: transform .2s;
}
.toggle.on::after { transform: translateX(20px); }

/* ── Main layout ── */
.main {
  display: grid;
  grid-template-columns: auto 1fr;
  grid-template-rows: auto auto;
  gap: 20px;
  max-width: 1100px;
  margin: 0 auto;
  align-items: start;
}

/* ── Video ── */
.video-wrap {
  grid-row: 1 / 3;
  border-radius: 14px; overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--card-sh);
}
.video-wrap img { display: block; width: 520px; height: 390px; object-fit: cover; }

/* ── Right column ── */
.right { display: flex; flex-direction: column; gap: 16px; }

/* ── Top-right: detection + word side by side ── */
.top-right { display: flex; gap: 16px; }

.card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px;
  box-shadow: var(--card-sh);
}
.clabel { font-size: .67rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text3); margin-bottom: 8px; }

/* Detection card */
.detect-card { flex: 0 0 160px; }
#big-letter { font-size: 4.5rem; font-weight: 700; line-height: 1; color: var(--text); text-align: center; }

/* Bars */
.bar-bg { background: var(--bg3); border-radius: 5px; height: 7px; overflow: hidden; margin-top: 5px; }
.bar-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg,#3b82f6,#06b6d4); transition: width .12s ease; }
#conf-val { font-size: .8rem; color: var(--text2); margin-top: 4px; }
.hold-bg { background: var(--bg3); border-radius: 5px; height: 9px; overflow: hidden; margin-top: 5px; }
.hold-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg,#f59e0b,#ef4444); transition: width .08s linear; }

/* Word card */
.word-card { flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
#current-word {
  font-size: 2rem; font-weight: 700; letter-spacing: .08em;
  color: var(--text); min-height: 44px; word-break: break-all;
  padding: 4px 0;
}
#current-word.empty { color: var(--text3); font-size: 1rem; font-weight: 400; padding-top: 10px; }

.btn-row { display: flex; gap: 8px; margin-top: 10px; }
button {
  flex: 1; padding: 8px 10px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg3);
  color: var(--text2); font-size: .8rem; cursor: pointer; transition: background .15s, color .15s;
}
button:hover { background: var(--border); color: var(--text); }
button.primary { background: #3b82f6; border-color: #3b82f6; color: #fff; }
button.primary:hover { background: #2563eb; }

/* ── Sentence box ── */
.sentence-card { grid-column: 2; }
#sentence-box {
  background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; font-size: 1.25rem; font-weight: 500;
  min-height: 64px; word-break: break-word; white-space: pre-wrap;
  color: var(--text); letter-spacing: .03em; line-height: 1.5;
  margin-bottom: 10px;
}
#sentence-box.empty { color: var(--text3); font-style: italic; font-weight: 400; font-size: 1rem; }

.hint { font-size: .69rem; color: var(--text3); line-height: 1.7; margin-top: 4px; }

/* ── Guidance hint ── */
.hint-banner {
  display: none;
  align-items: center; gap: 8px;
  background: #f59e0b22; border: 1px solid #f59e0b66;
  border-radius: 8px; padding: 8px 14px;
  font-size: .82rem; font-weight: 600; color: #f59e0b;
  margin-top: 4px;
}
.hint-banner.visible { display: flex; }
.hint-banner svg { flex-shrink: 0; }
</style>
</head>
<body>

<div class="topbar">
  <h1>ASL Live Recognition</h1>
  <div class="toggle-wrap">
    <span class="toggle-label" id="theme-label">Dark</span>
    <div class="toggle" id="theme-toggle" onclick="toggleTheme()"></div>
  </div>
</div>

<div class="main">

  <!-- Video -->
  <div class="video-wrap">
    <img src="/video_feed" alt="feed">
  </div>

  <!-- Right column -->
  <div class="right">

    <!-- Detection + Current Word -->
    <div class="top-right">

      <!-- Detection card -->
      <div class="card detect-card">
        <div class="clabel">Detected Sign</div>
        <div id="big-letter">—</div>
        <div style="margin-top:12px">
          <div class="clabel">Confidence</div>
          <div class="bar-bg"><div class="bar-fill" id="conf-bar" style="width:0%"></div></div>
          <div id="conf-val">0%</div>
        </div>
        <div style="margin-top:10px">
          <div class="clabel">Hold Progress</div>
          <div class="hold-bg"><div class="hold-fill" id="hold-bar" style="width:0%"></div></div>
        </div>
      </div>

      <!-- Current word card -->
      <div class="card word-card">
        <div>
          <div class="clabel">Current Word</div>
          <div id="current-word" class="empty">Sign letters...</div>
        </div>
        <div class="btn-row">
          <button class="primary" onclick="addWord()" title="Add word to sentence (or sign SPACE)">Add Word</button>
          <button onclick="clearWord()" title="Clear current word">Clear</button>
        </div>
      </div>

    </div>

    <!-- Guidance Hint Banner -->
    <div class="hint-banner" id="hint-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span id="hint-text"></span>
    </div>

    <!-- Sentence -->
    <div class="card sentence-card">
      <div class="clabel">Sentence</div>
      <div id="sentence-box" class="empty">Your sentence will appear here...</div>
      <div class="btn-row">
        <button onclick="clearAll()">Clear All</button>
        <button onclick="clearSentence()">Clear Sentence</button>
      </div>
      <div class="hint" style="margin-top:8px">
        Sign letters to build a word · <strong>ADD WORD</strong> commits word to sentence ·
        <strong>SPACE</strong> inserts a space character ·
        <strong>DELETE</strong> removes last letter · <strong>NOTHING</strong> pauses input
      </div>
    </div>

  </div>
</div>

<script>
const THRESH = 0.85;
let darkMode = true;

function toggleTheme() {
  darkMode = !darkMode;
  document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  document.getElementById('theme-label').textContent = darkMode ? 'Dark' : 'Light';
  document.getElementById('theme-toggle').classList.toggle('on', darkMode);
}
// Start with toggle in dark (on) state
document.getElementById('theme-toggle').classList.add('on');

async function poll() {
  try {
    const d = await fetch('/state').then(r => r.json());

    // Detected sign
    const ok = d.confidence >= THRESH && d.label && !['NOTHING','nothing','Nothing'].includes(d.label);
    document.getElementById('big-letter').textContent = ok ? d.label : '—';

    // Bars
    const pct = Math.round(d.confidence * 100);
    document.getElementById('conf-bar').style.width = pct + '%';
    document.getElementById('conf-val').textContent  = pct + '%';
    document.getElementById('hold-bar').style.width  = Math.round(d.progress * 100) + '%';

    // Hint banner
    const hintBanner = document.getElementById('hint-banner');
    const hintText   = document.getElementById('hint-text');
    if (d.hint) {
      hintText.textContent = d.hint;
      hintBanner.classList.add('visible');
    } else {
      hintBanner.classList.remove('visible');
    }

    // Current word
    const wbox = document.getElementById('current-word');
    if (!d.current_word) {
      wbox.textContent = 'Sign letters...';
      wbox.classList.add('empty');
    } else {
      wbox.textContent = d.current_word;
      wbox.classList.remove('empty');
    }

    // Sentence
    const sbox = document.getElementById('sentence-box');
    if (!d.sentence) {
      sbox.textContent = 'Your sentence will appear here...';
      sbox.classList.add('empty');
    } else {
      sbox.textContent = d.sentence;
      sbox.classList.remove('empty');
    }
  } catch(_) {}
  setTimeout(poll, 100);
}

async function addWord()       { await fetch('/add_word',       {method:'POST'}); }
async function clearWord()     { await fetch('/clear_word',     {method:'POST'}); }
async function clearSentence() { await fetch('/clear_sentence', {method:'POST'}); }
async function clearAll()      { await fetch('/clear',          {method:'POST'}); }

poll();
</script>
</body>
</html>"""


def gen_frames():
    while True:
        with frame_lock:
            f = latest_frame
        if f is None:
            time.sleep(0.03); continue
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(dict(state))

@app.route('/add_word', methods=['POST'])
def add_word():
    with state_lock:
        w = state["current_word"].strip()
        if w:
            state["sentence"] = (state["sentence"] + " " + w).strip()
        state["current_word"] = ""
    _hold.update({"label": None, "streak": 0, "timer": None, "done": False})
    return ('', 204)

@app.route('/clear_word', methods=['POST'])
def clear_word():
    with state_lock:
        state["current_word"] = ""
    _hold.update({"label": None, "streak": 0, "timer": None, "done": False})
    return ('', 204)

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    with state_lock:
        state["sentence"] = ""
    return ('', 204)

@app.route('/clear', methods=['POST'])
def clear():
    with state_lock:
        state["current_word"] = ""
        state["sentence"]     = ""
        state["progress"]     = 0.0
    _hold.update({"label": None, "streak": 0, "timer": None, "done": False})
    return ('', 204)

if __name__ == '__main__':
    print("\n=== ASL Live Demo — http://localhost:5000 ===\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
