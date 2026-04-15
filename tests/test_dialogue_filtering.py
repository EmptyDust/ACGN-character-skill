from types import SimpleNamespace

from tools.dialogue_extractor import DialogueExtractor


def _make_extractor(require_dialogue_quote: bool = False, skip_non_dialogue_events: bool = False):
    extractor = DialogueExtractor.__new__(DialogueExtractor)
    extractor.require_dialogue_quote = require_dialogue_quote
    extractor.skip_non_dialogue_events = skip_non_dialogue_events
    extractor.special_speakers = {}
    extractor._speaker_extractor = SimpleNamespace(normalize_speaker=lambda name: {"鱼住哲雄": "鱼住"}.get(name, name))
    return extractor


def test_parse_speaker_from_text_extracts_known_speaker_and_keeps_quote() -> None:
    extractor = _make_extractor(require_dialogue_quote=True)
    event = SimpleNamespace(text="鱼住哲雄 「嗯，是和被害人住在一起的儿子和儿媳。」", confidence=0.98)

    speaker, conf = extractor._parse_speaker_from_text(event, {"鱼住哲雄", "冬子"})

    assert speaker == "鱼住"
    assert conf == 0.98
    assert event.text == "「嗯，是和被害人住在一起的儿子和儿媳。」"


def test_parse_speaker_from_text_rejects_non_dialogue_when_quote_required() -> None:
    extractor = _make_extractor(require_dialogue_quote=True)
    event = SimpleNamespace(text="我把烟灰弹进烟缸。", confidence=0.95)

    speaker, conf = extractor._parse_speaker_from_text(event, {"鱼住", "冬子"})

    assert speaker is None
    assert conf == 0.0
    assert event.text == "我把烟灰弹进烟缸。"


def test_should_skip_event_when_dialogue_requires_speaker_and_quote() -> None:
    extractor = _make_extractor(require_dialogue_quote=True, skip_non_dialogue_events=True)
    event = SimpleNamespace(text="我把烟灰弹进烟缸。")

    assert extractor._should_skip_event(event, speaker=None) is True
    assert extractor._should_skip_event(SimpleNamespace(text="「嗯，是这样。」"), speaker=None) is True
    assert extractor._should_skip_event(SimpleNamespace(text="「嗯，是这样。」"), speaker="鱼住") is False
