# Realistic camera service: decoder threads keep the latest frame per stream, one batched YOLO call per tick, per-camera frame accounting.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Servico de cameras REALISTA: N threads de decode (cv2, abertura em paralelo) mantem o ULTIMO quadro de cada stream; um laco de inferencia em lote
roda a cada 1/TARGET s sobre os quadros NOVOS (contador por stream), YOLO em lote (model.predict). Conta quadros UNICOS processados e descartados.
uso: python batchsvc.py <urls.txt> <dur_s> <target_fps> <model> <dev> """
import sys, time, json, threading
import cv2
import psutil
from ultralytics import YOLO

urls = [l.strip() for l in open(sys.argv[1]) if l.strip()]
dur, target, model, dev = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
dev = "cpu" if dev == "cpu" else int(dev)
N = len(urls)
last = [None] * N
cnt = [0] * N          # quadros decodificados (unicos) por stream
opened = [False] * N
stop = False


def reader(i):
    cap = cv2.VideoCapture(urls[i])
    opened[i] = cap.isOpened()
    while not stop and cap.isOpened():
        ok, fr = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        last[i] = fr
        cnt[i] += 1
    cap.release()


t_open0 = time.time()
ths = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(N)]
for t in ths:
    t.start()
while time.time() - t_open0 < 120 and sum(1 for i in range(N) if last[i] is not None) < N:
    time.sleep(0.5)
n_ready = sum(1 for i in range(N) if last[i] is not None)
open_s = time.time() - t_open0
m = YOLO(model)
m.predict([last[i] for i in range(N) if last[i] is not None][:2], imgsz=640, device=dev, half=(dev != "cpu"), verbose=False)  # aquecimento
seen = [0] * N
processed, dropped, lat = 0, 0, []
psutil.cpu_percent(interval=None)
t0 = time.time()
cpu = []
period = 1.0 / target
nxt = t0
while time.time() - t0 < dur:
    batch, idx = [], []
    for i in range(N):
        if last[i] is not None and cnt[i] > seen[i]:
            batch.append(last[i]); idx.append(i)
    if batch:
        s = time.time()
        m.predict(batch, imgsz=640, device=dev, half=(dev != "cpu"), verbose=False)
        lat.append(time.time() - s)
        for i in idx:
            dropped += cnt[i] - seen[i] - 1  # decodificados e nunca inferidos (so o ultimo entra no lote)
            seen[i] = cnt[i]
        processed += len(batch)
    nxt += period
    sl = nxt - time.time()
    if sl > 0:
        time.sleep(sl)
    else:
        nxt = time.time()
    if len(lat) % 10 == 0:
        cpu.append(psutil.cpu_percent(interval=None))
stop = True
el = time.time() - t0
lat.sort()
print(json.dumps({"streams": N, "ready": n_ready, "open_s": round(open_s, 1), "processed_frames": processed, "secs": round(el, 1),
                  "fps_per_stream_inferred": round(processed / N / el, 2), "batch_p50_ms": round(lat[len(lat) // 2] * 1000, 1) if lat else None,
                  "batch_p95_ms": round(lat[int(len(lat) * 0.95) - 1] * 1000, 1) if lat else None, "decoded_not_inferred": dropped,
                  "cpu_pct_mean": round(sum(cpu) / len(cpu), 1) if cpu else None}))
