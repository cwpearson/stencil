#include "stencil/timer.hpp"

double Timer::get_elapsed() {
  pause();
  return elapsed_.count();
}

void Timer::clear() {
  pause();
  elapsed_ = Duration(0);
}

namespace timers {
/* extern */ Timer cudaRuntime;
}