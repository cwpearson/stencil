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
  void pause();
  void resume();
  double get_elapsed();
  Timer() : elapsed_(0), paused_(true) {}
};



namespace timers {
  extern Timer cudaRuntime;
}