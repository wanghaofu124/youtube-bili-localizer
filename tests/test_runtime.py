from yblocalizer.runtime import CancellationToken, PipelineContext, PipelineStage


def test_cancellation_tokens_are_independent() -> None:
    first, second = CancellationToken(), CancellationToken()
    first.cancel()
    assert first.cancelled is True
    assert second.cancelled is False


def test_pipeline_context_emits_structured_event() -> None:
    events = []
    context = PipelineContext(on_event=events.append)
    context.emit(PipelineStage.TRANSLATING, 62, "正在翻译")
    assert events[0].stage is PipelineStage.TRANSLATING
    assert events[0].progress == 62
