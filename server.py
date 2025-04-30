# app.py
from flask import Flask, render_template, jsonify
import paho.mqtt.client as mqtt
import json
import time
import numpy as np   # numpy 추가

app = Flask(__name__)

# MQTT 설정
broker = "monetgpu1.duckdns.org"
port   = 1883
topic  = "esp32uwb/#"

latest_data = {}

# 4개 앵커의 물리적 좌표 (단위: m)
anchor_positions = {
    "ANC7": (4.3,  0.0),
    "ANC2": (4.3,  5.85),
    "ANC4": (11.55,3.0),
    "ANC6": (7.05, 0.0)
}

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        aid  = data.get("anchor_id")
        if aid in anchor_positions:
            latest_data[aid] = {
                "distance": float(data.get("distance")),
                "timestamp": time.time()
            }
    except Exception:
        pass

client = mqtt.Client()
client.on_message = on_message

def connect_mqtt():
    while True:
        try:
            client.connect(broker, port)
            client.subscribe(topic)
            client.loop_start()
            break
        except Exception:
            time.sleep(5)

connect_mqtt()

# — 칼만 필터 정의 —
class KalmanFilter:
    def __init__(self, trust=0.5):
        self.value = None
        self.trust = trust
    def update(self, measured):
        if self.value is None:
            self.value = measured
        else:
            self.value += self.trust * (measured - self.value)
        return self.value

kf_x = KalmanFilter()
kf_y = KalmanFilter()

def compute_tag_position():
    # 모든 앵커 데이터가 있을 때만 계산
    if not all(a in latest_data for a in anchor_positions):
        return None
    anchors   = list(anchor_positions.keys())
    distances = [ latest_data[a]["distance"] for a in anchors ]
    coords    = [ anchor_positions[a] for a in anchors ]

    # 기준 앵커
    x0, y0 = coords[0];  r0 = distances[0]

    # Ax = b 형태로 변환
    A = []; b = []
    for (xi, yi), ri in zip(coords[1:], distances[1:]):
        A.append([ -2*(xi-x0), -2*(yi-y0) ])
        b.append( ri**2 - r0**2 - (xi**2 - x0**2) - (yi**2 - y0**2) )
    A = np.array(A);  b = np.array(b)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_raw, y_raw = sol[0], sol[1]

    # 칼만 필터 적용
    x_f = kf_x.update(x_raw)
    y_f = kf_y.update(y_raw)

    return {"x": x_f, "y": y_f}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    return jsonify({
        "anchors": latest_data,
        "tag":     compute_tag_position(),
        "anchor_positions": anchor_positions
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
