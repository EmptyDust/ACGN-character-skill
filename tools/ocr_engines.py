"""
OCR Engine Factory

Creates OCR functions for supported engines (paddleocr, easyocr, rapidocr).
Each function takes a PIL Image and returns (text, confidence).
"""

from typing import Callable
from PIL import Image


def create_ocr_func(engine: str) -> Callable[[Image.Image], tuple[str, float]]:
    """
    Create an OCR function for the specified engine.

    Args:
        engine: OCR engine name ("paddleocr", "easyocr", "rapidocr")

    Returns:
        Callable that takes a PIL Image and returns (text, confidence)
    """
    if engine == "paddleocr":
        return _create_paddleocr()
    elif engine == "easyocr":
        return _create_easyocr()
    elif engine == "rapidocr":
        return _create_rapidocr()
    else:
        raise ValueError(f"Unknown OCR engine: {engine}. Supported: paddleocr, easyocr, rapidocr")


def _create_paddleocr():
    import os
    os.environ.setdefault('FLAGS_call_stack_level', '2')
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('GLOG_minloglevel', '2')
    os.environ.setdefault('FLAGS_allocator_strategy', 'auto_growth')
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR not installed. Run: pip install paddleocr")

    ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

    def _from_legacy_result(result) -> tuple[str, float]:
        if result and result[0]:
            texts = []
            confidences = []
            for line in result[0]:
                texts.append(line[1][0])
                confidences.append(line[1][1])
            return (" ".join(texts), sum(confidences) / len(confidences))
        return ("", 0.0)

    def _from_predict_result(result) -> tuple[str, float]:
        if not result:
            return ("", 0.0)

        first = result[0]
        if isinstance(first, dict):
            texts = list(first.get("rec_texts") or [])
            confidences = list(first.get("rec_scores") or [])
            if texts:
                avg = sum(confidences) / len(confidences) if confidences else 0.0
                return (" ".join(texts), float(avg))
        return ("", 0.0)

    def ocr_func(image: Image.Image) -> tuple[str, float]:
        import numpy as np
        img_array = np.array(image)
        try:
            result = ocr.predict(img_array)
            return _from_predict_result(result)
        except TypeError:
            # Fall back to the older PaddleOCR API shape.
            pass

        result = ocr.ocr(img_array)
        return _from_legacy_result(result)

    return ocr_func


def _create_easyocr():
    try:
        import easyocr
    except ImportError:
        raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

    def ocr_func(image: Image.Image) -> tuple[str, float]:
        import numpy as np
        img_array = np.array(image)
        results = reader.readtext(img_array)
        if results:
            texts = [r[1] for r in results]
            confidences = [r[2] for r in results]
            return (" ".join(texts), sum(confidences) / len(confidences))
        return ("", 0.0)

    return ocr_func


def _create_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise ImportError("RapidOCR not installed. Run: pip install rapidocr-onnxruntime")

    ocr = RapidOCR()

    def ocr_func(image: Image.Image) -> tuple[str, float]:
        import numpy as np
        img_array = np.array(image)
        result, _ = ocr(img_array)
        if result:
            texts = [r[1] for r in result]
            confidences = [r[2] for r in result]
            return (" ".join(texts), sum(confidences) / len(confidences))
        return ("", 0.0)

    return ocr_func
