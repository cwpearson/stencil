#include <chrono>
#include <cmath>
#include <random>

#include "argparse/argparse.hpp"
#include "stencil/qap.hpp"

struct Mats {
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

  return Mats{
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
      d.at(i, j) = 1.0 / e;
    }
  }

  return Mats{
      .w = w,
      .d = d,
  };
}

Mat2D<double> blkdiag(int size,    // matrix size
                      double dMin, // diagonal entry range
                      double dMax,
                      double odMin, // off-diagonal range
                      double odMax, // off-diagonal range
                      int blkMin,   // block size
                      int blkMax) {

  Mat2D<double> m(size, size);

  std::uniform_real_distribution<double> offDiag(odMin, odMax);
  std::uniform_real_distribution<double> diag(dMin, dMax);
  std::uniform_int_distribution<int> blkSizeDist(blkMin, blkMax);
  std::default_random_engine re;

  int r = 0;
  while (r < size) {

    int blkSize = blkSizeDist(re);
    blkSize = min(blkSize, size - r);

    // fill the block
    for (int i = r; i < r + blkSize; ++i) {
      for (int j = r; j < r + blkSize; ++j) {
        m.at(i, j) = diag(re);
      }
    }
    // fill the right space
    for (int i = r; i < r + blkSize; ++i) {
      for (int j = r + blkSize; j < size; ++j) {
        m.at(i, j) = offDiag(re);
      }
    }
    // fill the bottom spkace
    for (int i = r + blkSize; i < size; ++i) {
      for (int j = r; j < r + blkSize; ++j) {
        m.at(i, j) = offDiag(re);
      }
    }

    r += blkSize;
  }

  return m;
}

Mats make_blkdiag(int s) {

  // 2x2 to 26x26 blocks of higher weight
  Mat2D<double> w = blkdiag(s, 100, 200, 10, 20, 2, 26);

  // 6x6 blocks of higher bandwidth
  Mat2D<double> d =
      blkdiag(s, 1.0 / 100.0, 1.0 / 64.0, 1.0 / 26.0, 1.0 / 25.0, 6, 6);

  return Mats{
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
      start = std::chrono::system_clock::now();

      for (int i = 0; i < nIters; ++i) {
        auto f = qap::solve(w, d, &cost);
      }

      elapsed =
          std::chrono::system_clock::now() - start;

      std::cout << " " << elapsed.count() / nIters << " " << cost;
    } else {
      std::cout << " "
                << "-"
                << " "
                << "-";
    }

    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;

  bench("blkdiag", make_blkdiag);
  bench("random", make_random);
  bench("matched", make_matched);

  return 0;
}
