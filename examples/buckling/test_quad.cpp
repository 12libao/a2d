#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <random>

#include "QuadLinearElastic.h"
#include "tools.h"

typedef double T;
typedef int I;

TEST(QuadLinearElastic, gemm) {
  // Set the seed values
  std::array<std::array<T, 3>, 3> A;
  std::array<std::array<T, 3>, 3> B;
  std::array<std::array<T, 3>, 3> C;

  // Set random values
  randomFill(A.begin()->begin(), A.end()->end(), -10.0, 10.0);
  randomFill(B.begin()->begin(), B.end()->end(), -10.0, 10.0);
  randomFill(C.begin()->begin(), C.end()->end(), -10.0, 10.0);

  tick("gemm");
  for (int i = 0; i < 100000000; ++i) {
    quad::gemm(A, B, C);
  }
  tock("gemm");

  print3x3Matrix("A", A);
  print3x3Matrix("B", B);
  print3x3Matrix("C", C);
}
