#pragma once

#include <chrono>

class Timer {

  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> Duration;
  typedef std::chrono::time_point<Clock> Time;

  Time start_;
  Duration elapsed_;
  bool paused_;

public:
  Timer() : elapsed_(0), paused_(true) {}

  // pause and reset timer
  void clear();

  inline void pause() {
    if (!paused_) {
      Duration elapsed = Clock::now() - start_;
      elapsed_ += elapsed;
      paused_ = true;
    }
  }

  inline void resume() {
    if (paused_) {
      start_ = Clock::now();
      paused_ = false;
    }
  }
  
  double get_elapsed();
};

namespace timers {
extern Timer cudaRuntime;
}

#if 0
#define CR_TIC() timers::cudaRuntime.resume()
#define CR_TOC() timers::cudaRuntime.pause()
#else
#define CR_TIC()
#define CR_TOC()
#endif