#!/bin/bash
# Service wrapper used by the sweeps.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# Cameras em UM processo com lote (ultralytics LoadStreams), sub-stream 640x360 e stream principal 1080p. Reusa mediamtx/ffmpeg do cameras.sh.
# env: OUT MODEL LEVELS_SUB ("64 128 192") LEVELS_MAIN ("32 64") DUR (45) TARGET_FPS (5) DEV (0|cpu)
OUT=${OUT:-out/cameras_svc}; MODEL=${MODEL:-yolo11m.pt}; LEVELS_SUB=${LEVELS_SUB:-"64 128 192"}; LEVELS_MAIN=${LEVELS_MAIN:-"32 64"}; DUR=${DUR:-45}; TARGET_FPS=${TARGET_FPS:-5}; DEV=${DEV:-0}
mkdir -p $OUT; OUT=$(realpath -m $OUT); mkdir -p /workspace/cam; cd /workspace/cam
log() { echo "$1" >> $OUT/cameras_svc.jsonl; }
export DEBIAN_FRONTEND=noninteractive
command -v ffmpeg >/dev/null || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; }
pip install -q ultralytics psutil >/dev/null 2>&1; pip install -q lapx >/dev/null 2>&1 || pip install -q lap >/dev/null 2>&1
if [ ! -x ./mediamtx ]; then
  URL=$(curl -s https://api.github.com/repos/bluenviron/mediamtx/releases/latest | python -c "import sys,json;a=json.load(sys.stdin)['assets'];print([x['browser_download_url'] for x in a if x['name'].endswith('linux_amd64.tar.gz')][0])")
  curl -sL "$URL" -o m.tgz && tar xzf m.tgz mediamtx
fi
[ -s raw.mp4 ] || { curl -sL -o raw.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 || true; }
[ -s raw.mp4 ] || ffmpeg -loglevel error -f lavfi -i testsrc2=size=1920x1080:rate=15 -t 30 -c:v libx264 -preset veryfast raw.mp4
[ -s main.mp4 ] || ffmpeg -loglevel error -y -i raw.mp4 -vf "scale=1920:1080,fps=15" -c:v libx264 -preset veryfast -an main.mp4
[ -s sub.mp4 ] || ffmpeg -loglevel error -y -i raw.mp4 -vf "scale=640:360,fps=15" -c:v libx264 -preset veryfast -an sub.mp4
printf 'paths:\n  all_others:\n' > mediamtx.yml
(./mediamtx mediamtx.yml > mtx.log 2>&1 &); sleep 3
log "{\"kind\":\"setup\",\"nproc\":$(nproc),\"model\":\"$MODEL\",\"dev\":\"$DEV\",\"target_fps\":$TARGET_FPS}"
cp /workspace/eco/batchsvc.py bworker.py 2>/dev/null || cp /batchsvc.py bworker.py
python -c "from ultralytics import YOLO; YOLO('$MODEL')" >/dev/null 2>&1
run_levels() { # tipo arquivo niveis
  for N in $3; do
    pkill -f "ffmpeg.*rtsp://127.0.0.1" 2>/dev/null; sleep 2
    : > streams.streams
    for i in $(seq 1 $N); do
      (ffmpeg -loglevel error -re -stream_loop -1 -i $2 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/c$i >/dev/null 2>&1 &)
      echo "rtsp://127.0.0.1:8554/c$i" >> streams.streams
    done
    sleep 10
    R=$(python bworker.py streams.streams $DUR $TARGET_FPS $MODEL $DEV 2>&1 | grep -a '^{' | tail -1)
    echo "$R" | python -c "import sys,json; d=json.loads(sys.stdin.read() or '{}'); d['kind']='batch'; d['feed']='$1'; print(json.dumps(d))" | tee -a $OUT/cameras_svc.jsonl
    nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | sed "s/^/VRAM $1 N=$N: /"
    FPS=$(echo "$R" | python -c "import sys,json; d=json.loads(sys.stdin.read() or '{}'); print(d.get('fps_per_stream_inferred',0))")
    python -c "import sys; sys.exit(0 if float('$FPS') >= 0.9*$TARGET_FPS else 1)" || break
  done
}
run_levels sub sub.mp4 "$LEVELS_SUB"
run_levels main main.mp4 "$LEVELS_MAIN"
pkill -f "ffmpeg.*rtsp://127.0.0.1" 2>/dev/null
echo CAMERAS_SVC_DONE
