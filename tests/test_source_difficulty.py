from yblocalizer.source_difficulty import band_for, common_words, score_english_text


def test_common_list_keeps_everyday_words_and_drops_fish_names() -> None:
    words = common_words()
    assert len(words) == 20000
    assert {"tuna", "bait", "jerk", "gaze", "shallow"} <= words
    assert "snapper" not in words
    assert "pompano" not in words


def test_two_common_words_do_not_score() -> None:
    text = " ".join(["I just got a jerk bait from the store today."] * 12)
    hint = score_english_text(text)
    assert hint is not None
    assert hint["rare_count"] == 0
    assert hint["score"] == 0
    assert hint["band"] == 0
    assert hint["advisory"] is True


def test_gazed_stems_to_a_common_word() -> None:
    text = " ".join(["He gazed across the bright open water and smiled."] * 12)
    hint = score_english_text(text)
    assert hint is not None
    assert hint["rare_count"] == 0


def test_repeated_specialized_word_counts_once() -> None:
    text = " ".join(["The snapper is on the boat again right now."] * 12)
    hint = score_english_text(text)
    assert hint is not None
    assert hint["rare_count"] == 1
    assert hint["score"] == 2
    assert hint["band"] == 0


def test_long_clause_with_two_rare_words_adds_sentence_points() -> None:
    filler = "people stood nearby and watched the water "
    sentence = (
        "The barracuda stayed by the jetty because the pompano kept circling the rocks "
        + filler
    )
    text = " ".join([sentence] * 4)
    hint = score_english_text(text)
    assert hint is not None
    assert hint["rare_count"] >= 2
    assert hint["score"] >= hint["rare_count"] * 2
    assert hint["score"] > hint["rare_count"] * 2


def test_copy_names_api_and_local_open_source_models() -> None:
    easy = score_english_text(" ".join(["I went to the store and bought some bread today."] * 12))
    hard = score_english_text(" ".join(f"zyxq{'a' * (i + 1)}" for i in range(30)) + " " + " ".join(["the"] * 50))
    assert easy is not None and hard is not None
    assert easy["band"] == 0
    assert hard["band"] == 60
    for hint in (easy, hard):
        assert "API" in hint["message"]
        assert "本地部署的开源模型" in hint["message"]
        assert "建议" in hint["message"]
        assert "在线接口" not in hint["message"]
        assert not hint["message"].startswith("用 API")
        assert not hint["message"].startswith("用本地部署的开源模型")
    assert easy["message"].startswith("这段大多是常见词。建议用本地部署的开源模型")
    assert hard["message"].startswith("术语多，本地部署的开源模型容易把专名译错。建议用 API。")


def test_short_text_is_not_scored() -> None:
    assert score_english_text("Hello there.") is None


def test_bands_step_every_twenty_points() -> None:
    assert [band_for(score) for score in (0, 19, 20, 39, 40, 59, 60, 79, 80, 90)] == [
        0, 0, 20, 20, 40, 40, 60, 60, 80, 80,
    ]
