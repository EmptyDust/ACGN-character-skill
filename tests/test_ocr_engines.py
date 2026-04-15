import pytest
from unittest.mock import patch

from PIL import Image

from tools.ocr_engines import create_ocr_func


class _PredictStyleOCR:
    def predict(self, _img):
        return [
            {
                "rec_texts": ["鱼住", "「嗯，是这样。"],
                "rec_scores": [0.9, 0.8],
            }
        ]

    def ocr(self, _img):
        raise AssertionError("legacy ocr() should not be used when predict() succeeds")


class _LegacyStyleOCR:
    def predict(self, _img):
        raise TypeError("predict API not supported")

    def ocr(self, _img):
        return [[
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("鱼住", 0.91)],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("「嗯，是这样。」", 0.89)],
        ]]


def test_create_paddleocr_supports_predict_style_results() -> None:
    with patch("paddleocr.PaddleOCR", return_value=_PredictStyleOCR()):
        fn = create_ocr_func("paddleocr")
        text, conf = fn(Image.new("RGB", (10, 10), "white"))

    assert text == "鱼住 「嗯，是这样。"
    assert conf == pytest.approx(0.85)


def test_create_paddleocr_falls_back_to_legacy_ocr_api() -> None:
    with patch("paddleocr.PaddleOCR", return_value=_LegacyStyleOCR()):
        fn = create_ocr_func("paddleocr")
        text, conf = fn(Image.new("RGB", (10, 10), "white"))

    assert text == "鱼住 「嗯，是这样。」"
    assert conf == pytest.approx(0.9)
