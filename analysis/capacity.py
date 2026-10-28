"""How many RTSP cameras one GPU sustains with YOLO11: process-per-camera against a batch service, the decoder ceiling, and where CPU runs out."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import BLUE, GREEN, ORANGE, RED, read, save_fig, save_table, update_summary

TARGET = 5.0   # inferred frames per second wanted per camera
PASS = 0.9     # a camera count passes when every camera gets at least 90% of the target

proc = read("cameras_process_per_stream.csv")
batch = read("cameras_rtsp_5090.csv")
nv = read("cameras_nvdec.csv")

proc["share_of_target"] = proc.fps_mean / TARGET
proc["workers_alive_share"] = proc.workers_ok / proc.streams
proc["valid"] = (proc.workers_alive_share == 1) & (proc.fps_min / TARGET >= PASS)
batch["share_of_target"] = batch.fps_per_stream_inferred / TARGET
batch["valid"] = batch.share_of_target >= PASS
save_table(proc.drop(columns=["err_sample"]).round(3), "process_per_camera", index=False)
save_table(batch.round(3), "batch_service", index=False)
save_table(nv.round(1), "nvdec_decode_ceiling", index=False)

max_proc = int(proc[proc.valid].streams.max())
max_batch_main = int(batch[(batch.feed == "main") & batch.valid].streams.max())
max_batch_sub = int(batch[(batch.feed == "sub") & batch.valid].streams.max()) if len(batch[(batch.feed == "sub") & batch.valid]) else 0

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pv = proc[proc.workers_alive_share == 1]
axes[0].plot(pv.streams, pv.fps_min, marker="o", color=ORANGE, label="process per camera (slowest camera)")
pi = proc[proc.workers_alive_share < 1]
axes[0].scatter(pi.streams, np.zeros(len(pi)), marker="X", s=70, color=RED, label="process per camera: workers died, no valid rate", zorder=5)
b = batch[batch.feed == "main"].sort_values("streams")
axes[0].plot(b.streams, b.fps_per_stream_inferred, marker="s", color=BLUE, label="batch service, main stream")
s = batch[batch.feed == "sub"].sort_values("streams")
axes[0].plot(s.streams, s.fps_per_stream_inferred, marker="^", color=GREEN, label="batch service, sub stream (640x360)")
axes[0].axhline(TARGET * PASS, ls="--", color="grey")
axes[0].text(8, TARGET * PASS + 0.1, "90% of target", fontsize=7, color="grey")
axes[0].set_xlabel("Cameras")
axes[0].set_ylabel("Inferred frames per second, per camera")
axes[0].legend(frameon=False, fontsize=7)
axes[1].bar([str(x) for x in proc.streams], proc.workers_ok, color=BLUE, label="started")
axes[1].bar([str(x) for x in proc.streams], proc.workers_err, bottom=proc.workers_ok, color=RED, label="died at start (CUDA allocation)")
axes[1].set_xlabel("Cameras (one process each)")
axes[1].set_ylabel("Worker processes")
axes[1].legend(frameon=False, fontsize=7)
valid_nv = nv[nv.parallel <= 32]
axes[2].plot(valid_nv.parallel, valid_nv.decode_fps_total, marker="o", color=BLUE)
axes[2].set_xlabel("Parallel hardware decoders")
axes[2].set_ylabel("Frames decoded per second, total")
axes[2].set_title("NVDEC saturates near 1,035 fps", fontsize=9)
save_fig(fig, "camera_capacity.png")

update_summary({"max_cameras_process_per_camera": max_proc, "max_cameras_batch_main": max_batch_main, "max_cameras_batch_sub_tested": max_batch_sub,
                "proc_32_fps_min": proc[proc.streams == 32].fps_min.iloc[0], "proc_48_workers_ok": int(proc[proc.streams == 48].workers_ok.iloc[0]),
                "proc_64_workers_ok": int(proc[proc.streams == 64].workers_ok.iloc[0]),
                "batch_main_32_share": batch[(batch.feed == "main") & (batch.streams == 32)].share_of_target.iloc[0],
                "batch_main_64_share": batch[(batch.feed == "main") & (batch.streams == 64)].share_of_target.iloc[0],
                "batch_sub_64_share": batch[(batch.feed == "sub") & (batch.streams == 64)].share_of_target.iloc[0],
                "batch_main_32_batch_p50_ms": batch[(batch.feed == "main") & (batch.streams == 32)].batch_p50_ms.iloc[0],
                "nvdec_ceiling_fps": valid_nv.decode_fps_total.max(), "nvdec_ceiling_streams_at_15fps": valid_nv.streams_equiv_at_input_fps.max(),
                "cpu_pct_proc_32": proc[proc.streams == 32].cpu_pct_mean.iloc[0], "cpu_pct_batch_main_32": batch[(batch.feed == "main") & (batch.streams == 32)].cpu_pct_mean.iloc[0]})
print(proc[["streams", "workers_ok", "fps_mean", "fps_min", "valid"]])
print(batch[["streams", "feed", "fps_per_stream_inferred", "valid"]])
