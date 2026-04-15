from tools.dialogue_extractor import _build_progress_desc


def test_build_progress_desc_formats_video_and_counts() -> None:
    line = _build_progress_desc("part01.mp4", 7)
    assert line.startswith("part01.mp4")
    assert "events:7" in line


def test_build_progress_desc_handles_zero_events() -> None:
    line = _build_progress_desc("part01.mp4", 0)
    assert line == "part01.mp4 events:0"
