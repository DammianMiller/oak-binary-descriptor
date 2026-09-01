"""Host-only tests for ThresholdAutotuner's convergent control loop.

The tuner needs depthai only for the config message it sends, so the queue is
faked and time is monkeypatched to drive the window/interval logic.
"""

import core.pipeline as pipeline
from core.pipeline import ThresholdAutotuner


class FakeQueue:
    def __init__(self):
        self.sent = []

    def send(self, cfg):
        self.sent.append(cfg)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def advance(self, dt):
        self.now += dt


def make_tuner(monkeypatch, target=400, **kwargs):
    clock = FakeClock()
    monkeypatch.setattr(pipeline.time, "time", clock.time)
    queue = FakeQueue()
    tuner = ThresholdAutotuner(queue, target, **kwargs)
    return tuner, queue, clock


def drive(tuner, clock, n_features, frames=20, dt=0.8):
    """Feed one window of frames, then advance past the update interval."""
    for _ in range(frames):
        tuner.update(n_features)
    clock.advance(dt)
    tuner.update(n_features)


def test_converges_upward_from_below_target(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400)
    # 800 features vs target 400: threshold must rise (fewer features).
    drive(tuner, clock, 800)
    assert tuner.threshold > 200.0
    assert len(queue.sent) == 1


def test_converges_downward_when_starved(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400)
    drive(tuner, clock, 100)
    assert tuner.threshold < 200.0
    assert len(queue.sent) == 1


def test_deadband_holds_near_target(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400)
    drive(tuner, clock, 396)  # within +/-5% of 400
    assert tuner.threshold == 200.0
    assert queue.sent == []


def test_step_is_sqrt_damped_and_clamped(monkeypatch):
    tuner, _, clock = make_tuner(monkeypatch, target=400)
    drive(tuner, clock, 1600)  # ratio 4 -> sqrt 2 -> clamped to 1.6
    assert abs(tuner.threshold - 320.0) < 1e-6


def test_dark_scene_drops_fast(monkeypatch):
    tuner, _, clock = make_tuner(monkeypatch, target=400)
    drive(tuner, clock, 0)
    assert tuner.threshold == 100.0  # 200 * 0.5


def test_clamps_at_bounds_and_stops_sending(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400, low=15.0)
    for _ in range(20):
        drive(tuner, clock, 1)  # keep demanding more features
    assert tuner.threshold == 15.0
    sends = len(queue.sent)
    drive(tuner, clock, 1)
    assert len(queue.sent) == sends  # clamped: no further updates


def test_target_setter_retargets(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400)
    tuner.target = 600
    drive(tuner, clock, 400)  # now below target: threshold must fall
    assert tuner.threshold < 200.0
    assert len(queue.sent) == 1


def feed(tuner, clock, n, fps, windows=1):
    for _ in range(windows):
        for _ in range(20):
            tuner.update(n, fps=fps)
        clock.advance(0.8)
        tuner.update(n, fps=fps)


def test_fps_guard_ratchets_floor_and_backs_off(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400, low=15.0)
    tuner._min_fps = 57.0  # 60 fps mode
    feed(tuner, clock, 390, 60.0)  # healthy, in deadband: no send
    assert tuner._fps_ok_seen
    assert queue.sent == []
    feed(tuner, clock, 100, 60.0)  # starved: count loop lowers the threshold
    lowered = tuner.threshold
    assert lowered < 200.0
    feed(tuner, clock, 320, 46.5)  # fps collapses right after: guard fires
    assert tuner.threshold > lowered
    assert tuner._low > 15.0
    # The ratcheted floor persists: starvation can no longer dive below it.
    floor = tuner._low
    feed(tuner, clock, 100, 60.0, windows=10)
    assert tuner.threshold >= floor


def test_fps_guard_ignores_environmentally_slow_stream(monkeypatch):
    tuner, _, clock = make_tuner(monkeypatch, target=400, low=15.0)
    tuner._min_fps = 57.0
    # Stream was never at 60 fps (dark scene, long exposure): the guard must
    # not fire, and the count loop keeps lowering the threshold as usual.
    feed(tuner, clock, 100, 41.0, windows=3)
    assert tuner.threshold < 200.0
    assert tuner._low == 15.0


def test_fps_guard_gives_up_when_raises_do_not_help(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400, low=15.0)
    tuner._min_fps = 57.0
    feed(tuner, clock, 390, 60.0)  # healthy once
    # Ping-pong: starved windows lower, slow windows trigger the guard.
    # After three unproductive raises the guard disables itself.
    for _ in range(6):
        feed(tuner, clock, 100, 41.0)
        if tuner._min_fps is None:
            break
    assert tuner._min_fps is None
    # Threshold stays bounded (no runaway to the ceiling).
    assert tuner.threshold < 1000.0


def test_interval_throttles_updates(monkeypatch):
    tuner, queue, clock = make_tuner(monkeypatch, target=400)
    for _ in range(20):
        tuner.update(800)  # fires on the 20th frame (first window, t=1000)
    assert len(queue.sent) == 1
    # 0.5 s later: a full window must not trigger (interval is 0.7 s).
    clock.advance(0.5)
    for _ in range(25):
        tuner.update(800)
    assert len(queue.sent) == 1
    # 1.0 s after the last send the next window does trigger.
    clock.advance(0.5)
    for _ in range(25):
        tuner.update(800)
    assert len(queue.sent) == 2
