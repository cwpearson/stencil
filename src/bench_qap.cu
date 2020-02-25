#include <chrono>
#include <cmath>

#include "stencil/argparse.hpp"
#include "stencil/qap.hpp"

struct Mats{
    Mat2D<double> w;
    Mat2D<double> d;
};

// generate a matrix of sxs
typedef Mats (*MatFunc)(int s);

Mats make_random(int s) {
    Mat2D<double> w(s, s);
    Mat2D<double> d(s, s);

    for (int i = 0; i < s; ++i) {
      for (int j = 0; j < s; ++j) {
        w.at(i, j) = rand();
        d.at(i, j) = rand();
      }
    }

    return Mats {
        .w = w,
        .d = d,
    };
}



Mats make_matched(int s) {
    Mat2D<double> w(s, s);
    Mat2D<double> d(s, s);

    for (int i = 0; i < s; ++i) {
      for (int j = 0; j < s; ++j) {
        double e = rand();
        w.at(i, j) = e;
        d.at(i, j) = 1.0/e;
      }
    }

    return Mats {
        .w = w,
        .d = d,
    };
}


void bench(const std::string &name, MatFunc func) {
    int nIters = 20;

    std::cout << name << "\n";
    std::cout << "size CRAFT(s) cost exact(s) cost\n";
    for (int s = 2; s < 40; ++s) {
  
      Mats wd = func(s);
  
      auto w = wd.w;
      auto d = wd.d;
  
      std::cout << s << " ";
  
      // benchmark CRAFT solution
      auto start = std::chrono::system_clock::now();
  
      double cost;
      for (int i = 0; i < nIters; ++i) {
        auto f = qap::solve_catch(w, d, &cost);
      }
  
      std::chrono::duration<double> elapsed =
          std::chrono::system_clock::now() - start;
  
      std::cout << elapsed.count() / nIters << " " << cost;
  
      // benchmark exact solution
      if (s < 9) {
        auto start = std::chrono::system_clock::now();
  
        double cost;
        for (int i = 0; i < nIters; ++i) {
          auto f = qap::solve(w, d, &cost);
        }
  
        std::chrono::duration<double> elapsed =
            std::chrono::system_clock::now() - start;
  
        std::cout << " " <<  elapsed.count() / nIters << " " << cost;
      } else {
        std::cout << " " << "-" << " " << "-";
      }
  
      std::cout << "\n";
    }
}


int main(int argc, char **argv) {

  bench("random", make_random);
  bench("matched", make_matched);

  return 0;
}
