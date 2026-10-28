# Data

Curated CSV files from the campaign (public Kaggle dataset `lucasrangelss/gpu-ai-benchmark-campaign-2026`). One row per measured run or attempt; the columns are described in the dataset page. Host locations and instance identifiers were removed.

- `cameras_process_per_stream.csv`: process-per-camera design: workers started, frames per second per camera, CPU, error sample
- `cameras_rtsp_5090.csv`: batch service on main and sub streams: frames per camera, batch time, CPU, frames decoded but not inferred
- `cameras_nvdec.csv`: hardware decoder saturation test
