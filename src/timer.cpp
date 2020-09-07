#include "stencil/timer.hpp"

void Timer::pause() {
  if (!paused_) {
    Duration elapsed = Clock::now() - start_;
    elapsed_ += elapsed;
    paused_ = true;
  }
}

void Timer::resume() {
  if (paused_) {
    start_ = Clock::now();
    paused_ = false;
  }
}

double Timer::get_elapsed() {
  pause();
  return elapsed_.count();
}

namespace timers {
  /* extern */ Timer cudaRuntime;
}