"""First-response latency: turn timing, sentence release, and dropped audio."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.context.audio_context_manager import AudioContextFrame, AudioContextManager
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from voiceos.core.contract import TurnSettings
from voiceos.runtime.pipeline import _turn_strategies, user_aggregator_params
from voiceos.runtime.services.speech_tts import SpeechServerTTSService
from voiceos.runtime.tools import faq_lookup_result
from voiceos.runtime.turns import TurnRecorder


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def push(r: TurnRecorder, clock: Clock, at: float, frame: Any) -> None:
    clock.t = at
    await r.on_push_frame(cast(FramePushed, SimpleNamespace(frame=frame)))


async def test_turn_timing_fields_are_recorded_without_the_transcript() -> None:
    clock = Clock()
    recorder = TurnRecorder(session_id="session-1")
    recorder._now = clock  # type: ignore[method-assign]
    await push(recorder, clock, 1.0, VADUserStartedSpeakingFrame())
    await push(recorder, clock, 2.0, VADUserStoppedSpeakingFrame())
    await push(
        recorder,
        clock,
        2.2,
        __import__("pipecat.frames.frames", fromlist=["TranscriptionFrame"]).TranscriptionFrame(
            "secret caller words", "caller", "2026-11-19T00:00:00Z"
        ),
    )
    await push(recorder, clock, 2.3, UserStoppedSpeakingFrame())
    await push(
        recorder,
        clock,
        2.5,
        __import__("pipecat.frames.frames", fromlist=["LLMTextFrame"]).LLMTextFrame("Hello."),
    )
    await push(recorder, clock, 2.7, TTSAudioRawFrame(b"\x00\x00", 24_000, 1))
    await push(recorder, clock, 2.8, BotStartedSpeakingFrame())
    await push(recorder, clock, 4.0, BotStoppedSpeakingFrame())

    [row] = recorder.timings
    assert row["session_id"] == "session-1"
    assert row["turn_id"] == 0
    assert row["speech_started_at"]
    assert row["speech_stopped_at"]
    assert row["transcript_final_at"]
    assert row["llm_first_token_at"]
    assert row["tts_first_audio_at"]
    assert row["bot_started_at"]
    assert row["bot_stopped_at"]
    assert row["total_response_ms"] == pytest.approx(800)
    assert "secret" not in str(row)


async def test_first_sentence_is_released_before_the_reply_finishes() -> None:
    aggregator = SimpleTextAggregator(aggregation_type=TextAggregationMode.SENTENCE)
    released: list[str] = []
    async for part in aggregator.aggregate("Hello. The rest of the reply is still arriving"):
        released.append(part.text)
    assert released == ["Hello."]
    assert "still arriving" in aggregator.text.text


class _Body:
    def __init__(self) -> None:
        self.second = False

    async def __aenter__(self) -> _Body:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def aiter_bytes(self, _chunk_size: int | None = None):
        yield b"\x01\x00"
        self.second = True
        yield b"\x02\x00"

    async def aread(self) -> bytes:
        return b""

    @property
    def status_code(self) -> int:
        return 200


class _Client:
    def __init__(self, body: _Body) -> None:
        self.body = body

    def stream(self, *_args: object, **_kwargs: object) -> _Body:
        return self.body

    async def aclose(self) -> None:
        return None


async def test_tts_yields_the_first_frame_before_the_rest_of_the_body() -> None:
    body = _Body()
    service = SpeechServerTTSService(base_url="http://speech/v1", voice="af_heart")
    await service._client.aclose()
    service._client = _Client(body)  # type: ignore[assignment]

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    service.start_tts_usage_metrics = _noop  # type: ignore[method-assign]
    service.stop_ttfb_metrics = _noop  # type: ignore[method-assign]

    frames = []
    async for frame in service.run_tts("Hello.", "ctx-1"):
        frames.append(frame)
        if len(frames) == 1:
            assert body.second is False
    assert body.second is True
    assert [frame.audio for frame in frames] == [b"\x01\x00", b"\x02\x00"]
    assert [frame.context_id for frame in frames] == ["ctx-1", "ctx-1"]


async def test_interruption_drops_queued_audio_and_keeps_order() -> None:
    manager = AudioContextManager()
    await manager.create("stale", interruptible=True)
    await manager.append("stale", TTSAudioRawFrame(b"\x01\x00", 24_000, 1))
    await manager.create("kept", interruptible=False)
    await manager.append("kept", TTSAudioRawFrame(b"\x02\x00", 24_000, 1))
    manager.reset()

    nxt = await asyncio.wait_for(manager.next_for_playback(), 0.2)
    assert isinstance(nxt, AudioContextFrame)
    assert nxt.context_id == "kept"
    played = await manager.get("kept", timeout=0.2)
    assert played.audio == b"\x02\x00"


def test_smart_turn_silence_matches_the_vad_window() -> None:
    agent = SimpleNamespace(turn=TurnSettings(mode="smart", vad_stop_secs=0.5))
    strategies = _turn_strategies(agent)  # type: ignore[arg-type]
    analyzer = strategies.stop[0]._turn_analyzer
    assert analyzer.params.stop_secs == 0.5
    assert agent.turn.vad_stop_secs == 0.5
    assert 0.4 <= agent.turn.vad_stop_secs <= 0.7


def test_faq_lookup_still_returns_the_answer() -> None:
    result = faq_lookup_result(
        [{"question": "What is your refund policy?", "answer": "Five working days."}],
        "What is your refund policy?",
    )
    assert result["matched"] is True
    assert result["answers"] == ["Five working days."]


async def test_tts_timeout_errors_without_a_long_silent_wait() -> None:
    class _TimeoutBody:
        async def __aenter__(self) -> _TimeoutBody:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def aiter_bytes(self, _chunk_size: int | None = None):
            await asyncio.sleep(0.05)
            raise httpx.ReadTimeout("read timed out")
            yield b""

        async def aread(self) -> bytes:
            return b""

        @property
        def status_code(self) -> int:
            return 200

    class _Client:
        def stream(self, *_args: object, **_kwargs: object) -> _TimeoutBody:
            return _TimeoutBody()

        async def aclose(self) -> None:
            return None

    service = SpeechServerTTSService(base_url="http://speech/v1", voice="af_heart", timeout_s=0.2)
    await service._client.aclose()
    service._client = _Client()  # type: ignore[assignment]

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    service.start_tts_usage_metrics = _noop  # type: ignore[method-assign]
    started = time.perf_counter()
    frames = [frame async for frame in service.run_tts("Hello.", "ctx-timeout")]
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0
    assert isinstance(frames[-1], ErrorFrame)
    assert "ReadTimeout" in str(frames[-1].error)


async def test_tts_interruption_closes_the_request() -> None:
    closed = asyncio.Event()

    class _Body:
        async def __aenter__(self) -> _Body:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            closed.set()

        async def aiter_bytes(self, _chunk_size: int | None = None):
            await asyncio.sleep(30)
            yield b"\x00\x00"

        async def aread(self) -> bytes:
            return b""

        @property
        def status_code(self) -> int:
            return 200

    class _Hanging:
        def stream(self, *_args: object, **_kwargs: object) -> _Body:
            return _Body()

        async def aclose(self) -> None:
            return None

    service = SpeechServerTTSService(base_url="http://speech/v1", voice="af_heart")
    await service._client.aclose()
    service._client = _Hanging()  # type: ignore[assignment]

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    service.start_tts_usage_metrics = _noop  # type: ignore[method-assign]

    async def drive() -> None:
        async for _frame in service.run_tts("Hello.", "ctx-cancel"):
            pass

    task = asyncio.create_task(drive())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), 1)


async def test_empty_final_transcript_closes_the_turn_and_the_next_turn_can_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = SimpleNamespace(turn=TurnSettings(mode="smart", vad_stop_secs=0.5))
    strategy = _turn_strategies(agent).stop[0]  # type: ignore[attr-defined]
    stopped: list[bool] = []

    async def trigger(**_kwargs: object) -> None:
        stopped.append(True)

    strategy.trigger_user_turn_stopped = trigger  # type: ignore[method-assign]
    blank = TranscriptionFrame("", "caller", "2026-11-19T00:00:00Z")
    blank.finalized = True
    await strategy._handle_transcription(blank)
    assert stopped == [True]
    params = user_aggregator_params(agent)  # type: ignore[arg-type]
    assert params.empty_user_turn is not None
    assert params.empty_user_turn.idle_prompt is None
    assert params.empty_user_turn.interrupted_prompt is None

    await strategy.handle_user_turn_stopped()
    assert strategy._transcript_finalized is False
    stopped.clear()
    await strategy._handle_transcription(blank)
    assert stopped == [True]

    recorder = TurnRecorder(session_id="session-9")
    with caplog.at_level("WARNING"):
        await push(recorder, Clock(), 1.0, blank)
    assert recorder._turns[0].user_text == []
    assert "empty final transcript" in caplog.text
    assert any(getattr(rec, "turn_id", None) == 0 for rec in caplog.records)
    assert any(getattr(rec, "session_id", None) == "session-9" for rec in caplog.records)


def test_both_transports_stay_in_the_contract() -> None:
    from typing import get_args

    from voiceos.core.contract import Transport

    assert set(get_args(Transport)) >= {"webrtc", "websocket"}
