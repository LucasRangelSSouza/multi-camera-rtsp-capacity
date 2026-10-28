import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_thirty_two_cameras_pass_in_both_designs():
    assert S["max_cameras_process_per_camera"] >= 32 and S["max_cameras_batch_main"] >= 32


def test_process_per_camera_collapses_at_48():
    assert S["proc_48_workers_ok"] == 1


def test_batch_service_misses_target_at_64_main_streams():
    assert S["batch_main_64_share"] < 0.5 < S["batch_sub_64_share"]


def test_decoder_ceiling_is_far_below_detector_ceiling():
    # the detector alone could serve hundreds of cameras at 5 fps; the decoder tops out near 69 streams of 15 fps video
    assert 60 < S["nvdec_ceiling_streams_at_15fps"] < 80
