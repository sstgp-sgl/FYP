// Playback state machine: idle/playing/paused/finished. Hard cuts only.
// Clock is monotonic (performance.now) and injectable for tests; the player
// owns no DOM and performs no network access.
export class Player {
  constructor({ now = () => performance.now(), onTick = () => {}, onShot = () => {}, onState = () => {} } = {}) {
    this.now = now;
    this.onTick = onTick;
    this.onShot = onShot;
    this.onState = onState;
    this.storyboard = null;
    this.state = "idle";       // idle | playing | paused | finished
    this.shotIndex = 0;
    this.elapsed = 0;          // seconds into the current shot
    this.baseTime = 0;         // monotonic anchor for the playing segment
    this.totalElapsed = 0;
    this.rafId = 0;
    this._frame = this._frame.bind(this);
  }

  load(storyboard, scene) {
    if (!storyboard || !scene) {
      this.state = "idle";
      this._emit();
      return;
    }
    if (storyboard.scene_id !== scene.scene_id) {
      throw new Error("分镜与场景不匹配（scene_id）");
    }
    if (storyboard.scene_revision !== scene.revision) {
      throw new Error("分镜是在旧场景版本上制作的，请在新版本上重新校验保存");
    }
    if (storyboard.asset_sha256 !== scene.asset_sha256) {
      throw new Error("分镜对应的点云资产与当前场景不一致");
    }
    if (!storyboard.shots || storyboard.shots.length === 0) {
      throw new Error("分镜没有镜头");
    }
    if (storyboard.validation && storyboard.validation.status !== "valid") {
      throw new Error("分镜校验未通过，不能播放");
    }
    this.storyboard = storyboard;
    this.stop();
  }

  play() {
    if (!this.storyboard) return;
    if (this.state === "finished") this._reset();
    if (this.state === "playing") return;
    this.state = "playing";
    this.baseTime = this.now() - this.elapsed * 1000;
    cancelAnimationFrame(this.rafId);
    this.rafId = requestAnimationFrame(this._frame);
    this._emitShot();
    this._emit();
  }

  pause() {
    if (this.state !== "playing") return;
    this.elapsed = this._shotElapsed();
    this.state = "paused";
    cancelAnimationFrame(this.rafId);
    this._emit();
  }

  stop() {
    cancelAnimationFrame(this.rafId);
    this._reset();
  }

  next() {
    if (!this.storyboard) return;
    const wasPlaying = this.state === "playing";
    cancelAnimationFrame(this.rafId);
    if (this.shotIndex < this.storyboard.shots.length - 1) {
      this.shotIndex += 1;
      this.elapsed = 0;
      this.state = wasPlaying ? "playing" : "paused";
      this._emitShot();
      if (wasPlaying) {
        this.baseTime = this.now();
        this.rafId = requestAnimationFrame(this._frame);
      }
    } else if (wasPlaying) {
      this._finish();
    }
    this._emit();
  }

  previous() {
    if (!this.storyboard) return;
    const wasPlaying = this.state === "playing";
    cancelAnimationFrame(this.rafId);
    if (this.shotIndex > 0) {
      this.shotIndex -= 1;
    }
    this.elapsed = 0;
    this.state = wasPlaying ? "playing" : "paused";
    this._emitShot();
    if (wasPlaying) {
      this.baseTime = this.now();
      this.rafId = requestAnimationFrame(this._frame);
    }
    this._emit();
  }

  replay() {
    if (!this.storyboard) return;
    this._reset();
    this.play();
  }

  _reset() {
    cancelAnimationFrame(this.rafId);
    this.shotIndex = 0;
    this.elapsed = 0;
    this.totalElapsed = 0;
    this.state = "idle";
    this._emitShot();
    this._emit();
  }

  _finish() {
    cancelAnimationFrame(this.rafId);
    this.state = "finished";
    this.elapsed = this.currentShot().duration_seconds;
    this._emit();
  }

  _shotElapsed() {
    return Math.min(this.elapsed + (this.now() - this.baseTime) / 1000,
      this.currentShot().duration_seconds);
  }

  _frame() {
    if (this.state !== "playing") return;
    const shot = this.currentShot();
    const nowMs = this.now();
    this.elapsed = (nowMs - this.baseTime) / 1000;
    if (this.elapsed >= shot.duration_seconds) {
      this.totalElapsed += shot.duration_seconds;
      if (this.shotIndex < this.storyboard.shots.length - 1) {
        this.shotIndex += 1;
        this.elapsed = 0;
        this.baseTime = nowMs;
        this._emitShot();
      } else {
        this._finish();
        return;
      }
    }
    this.onTick({
      shotIndex: this.shotIndex,
      shotElapsed: this.elapsed,
      shotDuration: shot.duration_seconds,
      totalElapsed: this.totalElapsed + this.elapsed,
    });
    this.rafId = requestAnimationFrame(this._frame);
  }

  currentShot() {
    return this.storyboard.shots[this.shotIndex];
  }

  _emitShot() {
    if (!this.storyboard) return;
    const shot = this.currentShot();
    this.onShot({
      index: this.shotIndex,
      shot,
      count: this.storyboard.shots.length,
      totalElapsed: this.totalElapsed + (this.state === "playing" ? this._shotElapsed() : this.elapsed),
    });
  }

  _emit() {
    this.onState(this.state);
  }
}
