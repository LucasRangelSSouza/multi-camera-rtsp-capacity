#!/bin/bash
# Starts MediaMTX and N looped RTSP streams and runs the process-per-camera sweep.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# Capacidade REAL de cameras: N streams RTSP 1080p simulados (mediamtx + ffmpeg -c copy em loop) -> deteccao+tracking por stream (ultralytics, vid_stride)
# e, separado, capacidade de decode NVDEC. Roda sobre qualquer maquina (GPU ou CPU). Saida: $OUT/cameras.jsonl
# env: OUT (default out/cameras) MODEL (default yolo11m.pt) LEVELS (default "8 16 32 48") DUR (default 60) TARGET_FPS (default 5) DEV (default 0; "cpu" p/ ensaio local)
OUT=${OUT:-out/cameras}; MODEL=${MODEL:-yolo11m.pt}; LEVELS=${LEVELS:-"8 16 32 48"}; DUR=${DUR:-60}; TARGET_FPS=${TARGET_FPS:-5}; DEV=${DEV:-0}
mkdir -p $OUT; OUT=$(realpath -m $OUT); mkdir -p /workspace/cam; cd /workspace/cam  # OUT absoluto: o script muda de diretorio
log() { echo "$1" >> $OUT/cameras.jsonl; }
export DEBIAN_FRONTEND=noninteractive
command -v ffmpeg >/dev/null || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; }
pip install -q ultralytics psutil >/dev/null 2>&1; pip install -q lapx >/dev/null 2>&1 || pip install -q lap >/dev/null 2>&1
# mediamtx
if [ ! -x ./mediamtx ]; then
  URL=$(curl -s https://api.github.com/repos/bluenviron/mediamtx/releases/latest | python -c "import sys,json;a=json.load(sys.stdin)['assets'];print([x['browser_download_url'] for x in a if x['name'].endswith('linux_amd64.tar.gz')][0])")
  curl -sL "$URL" -o m.tgz && tar xzf m.tgz mediamtx
fi
# video de amostra (pessoas/carros, H.264). Fallback: padrao sintetico (otimista: poucos objetos)
if [ ! -s sample.mp4 ]; then
  curl -sL -o sample.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 || true
  [ -s sample.mp4 ] || ffmpeg -loglevel error -f lavfi -i testsrc2=size=1920x1080:rate=15 -t 30 -c:v libx264 -preset veryfast sample.mp4
fi
# normaliza para 1080p15 H.264 (decode realista de camera 1080p); o conteudo continua sendo cena real de pessoas/carros
if [ ! -s sample1080.mp4 ]; then ffmpeg -loglevel error -y -i sample.mp4 -vf "scale=1920:1080,fps=15" -c:v libx264 -preset veryfast -an sample1080.mp4; fi
mv -f sample1080.mp4 sample.mp4 2>/dev/null; [ -s sample.mp4 ]
V=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate -of csv=p=0 sample.mp4)
IFPS=$(python -c "a,b=('$V'.split(',')[3]).split('/');print(round(int(a)/int(b)))")
STRIDE=$(python -c "print(max(1,round($IFPS/$TARGET_FPS)))")
log "{\"kind\":\"setup\",\"video\":\"$V\",\"input_fps\":$IFPS,\"stride\":$STRIDE,\"nproc\":$(nproc),\"model\":\"$MODEL\",\"dev\":\"$DEV\"}"
printf 'paths:
  all_others:
' > mediamtx.yml  # mediamtx >= 1.x exige o caminho declarado
(./mediamtx mediamtx.yml > mtx.log 2>&1 &); sleep 3
cat > worker.py <<'PY'
import sys, time, json
from ultralytics import YOLO
url, dur, stride, model, dev, sid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
m = YOLO(model)
dev = "cpu" if dev == "cpu" else int(dev)
n, t0, tfirst = 0, time.time(), None
try:
    for r in m.track(source=url, stream=True, vid_stride=stride, tracker="bytetrack.yaml", verbose=False, imgsz=640, device=dev, half=(dev != "cpu")):
        if tfirst is None:
            tfirst = time.time()
        n += 1
        if time.time() - t0 > dur:
            break
except Exception as e:
    print(json.dumps({"sid": sid, "error": repr(e)[:200]})); sys.exit(0)
el = time.time() - (tfirst or t0)
print(json.dumps({"sid": sid, "frames": n, "secs": round(el, 2), "fps": round(n / el, 2) if el > 0 else 0, "startup_s": round((tfirst or t0) - t0, 1)}))
PY
# aquece o modelo uma vez (baixa pesos)
python -c "from ultralytics import YOLO; YOLO('$MODEL')" >/dev/null 2>&1
for N in $LEVELS; do
  pkill -f "ffmpeg.*rtsp://127.0.0.1" 2>/dev/null; sleep 2
  for i in $(seq 1 $N); do
    (ffmpeg -loglevel error -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam$i >/dev/null 2>&1 &)
  done
  sleep 8
  CPU0=$(python -c "import psutil;print(psutil.cpu_percent(interval=None))")
  for i in $(seq 1 $N); do
    (python worker.py rtsp://127.0.0.1:8554/cam$i $DUR $STRIDE $MODEL $DEV $i > w_${N}_$i.out 2>&1 &)
  done
  python - <<PY
import time, subprocess, psutil, json
t0 = time.time(); cpu = []
while time.time() - t0 < $DUR + 40:
    time.sleep(5); cpu.append(psutil.cpu_percent(interval=None))
    out = subprocess.run("pgrep -f worker.py | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
    if out == "0": break
open("cpu_$N.json", "w").write(json.dumps(cpu))
PY
  python - <<PY
import glob, json, statistics as st
res = []
for f in glob.glob("w_*.out"):
    for line in open(f):
        line = line.strip()
        if line.startswith("{") and '"sid"' in line:
            res.append(json.loads(line))
ok = [r for r in res if "fps" in r]
err = [r for r in res if "error" in r]
cpu = json.load(open("cpu_$N.json")) if __import__("os").path.exists("cpu_$N.json") else []
rec = {"kind": "level", "streams": $N, "workers_ok": len(ok), "workers_err": len(err), "err_sample": (err[0]["error"] if err else None),
       "fps_mean": round(st.mean(r["fps"] for r in ok), 2) if ok else None, "fps_min": min((r["fps"] for r in ok), default=None), "fps_agg": round(sum(r["fps"] for r in ok), 1),
       "target_fps": $TARGET_FPS, "cpu_pct_mean": round(st.mean(cpu), 1) if cpu else None}
print(json.dumps(rec))
open("$OUT/cameras.jsonl", "a").write(json.dumps(rec) + "\n")
PY
  rm -f w_*.out
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | sed "s/^/GPU_AFTER N=$N: /"
  # criterio de parada: per-stream abaixo de 90% do alvo
  STOP=$(python - <<PY
import json
l=[json.loads(x) for x in open("$OUT/cameras.jsonl") if '"level"' in x][-1]
print(1 if (l["fps_mean"] or 0) < 0.9*$TARGET_FPS or l["workers_err"] > 0 else 0)
PY
)
  [ "$STOP" = 1 ] && break
done
pkill -f "ffmpeg.*rtsp://127.0.0.1" 2>/dev/null
# NVDEC puro (so GPU NVIDIA): decode do arquivo N vezes em paralelo; capacidade = fps total / fps de entrada
if command -v nvidia-smi >/dev/null && [ "$DEV" != "cpu" ]; then
  for N in 8 16 32 64; do
    S=$(date +%s.%N)
    for i in $(seq 1 $N); do (ffmpeg -loglevel error -hwaccel cuda -stream_loop 4 -i sample.mp4 -f null - >/dev/null 2>&1 &); done
    while pgrep -x ffmpeg >/dev/null; do sleep 1; done
    E=$(date +%s.%N)
    FR=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 sample.mp4)
    python - <<PY
import json
secs = $E - $S; frames = $FR * 5 * $N
rec = {"kind": "nvdec", "parallel": $N, "frames_total": frames, "secs": round(secs, 1), "decode_fps_total": round(frames / secs, 1), "streams_equiv_at_input_fps": round(frames / secs / $IFPS, 1)}
print(json.dumps(rec)); open("$OUT/cameras.jsonl", "a").write(json.dumps(rec) + "\n")
PY
  done
fi
echo CAMERAS_DONE
