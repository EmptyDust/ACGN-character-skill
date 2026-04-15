from textwrap import dedent

from tools.extract_character_lines import (
    extract_cn_utterances,
    extract_merged_utterances,
    pair_utterances_by_file_and_order,
)


MERGED_SAMPLE = dedent(
    """
    ====================
    FILE NAME: sample_a.txt
    ====================

    ＃玲人
    「――え？」
    ＃冬子
    「私の名前、さ」
    そう彼女――冬子は言うと、くるりと己に背を向けた。
    ＃冬子
    「聞いてる？」
    ＃玲人
    「ああ、勿論」
    """
).strip()


CN_SAMPLE = dedent(
    """
    ====================
    FILE NAME: sample_a.txt
    ====================

    玲人
    “——诶？”
    冬子
    「我的名字是」
    是的，她——冬子这样说着，便转过身背对着我。
    冬子
    “你在听吗？”
    玲人
    "啊，当然"
    """
).strip()


def test_extract_merged_utterances_only_takes_target_speaker_lines() -> None:
    utterances = extract_merged_utterances(MERGED_SAMPLE, {"冬子", "朽木冬子"})

    assert [u.file_name for u in utterances] == ["sample_a.txt", "sample_a.txt"]
    assert [u.order for u in utterances] == [1, 2]
    assert [u.text for u in utterances] == ["「私の名前、さ」", "「聞いてる？」"]


def test_extract_cn_utterances_uses_standalone_speaker_lines() -> None:
    utterances = extract_cn_utterances(CN_SAMPLE, {"冬子", "朽木冬子"})

    assert [u.file_name for u in utterances] == ["sample_a.txt", "sample_a.txt"]
    assert [u.order for u in utterances] == [1, 2]
    assert [u.text for u in utterances] == ["「我的名字是」", "“你在听吗？”"]


def test_pair_utterances_by_file_and_order_aligns_merged_and_cn() -> None:
    merged = extract_merged_utterances(MERGED_SAMPLE, {"冬子", "朽木冬子"})
    cn = extract_cn_utterances(CN_SAMPLE, {"冬子", "朽木冬子"})

    paired = pair_utterances_by_file_and_order(merged, cn)

    assert len(paired) == 2
    assert paired[0]["file_name"] == "sample_a.txt"
    assert paired[0]["speaker"] == "朽木冬子"
    assert paired[0]["text_ja"] == "「私の名前、さ」"
    assert paired[0]["text_zh"] == "「我的名字是」"
