#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include "QuadLinearElastic.h"
#include "tools.h"

int main() {
  typedef double T;
  T dh = 1e-30;

  // Set the seed values
  std::array<std::array<T, 3>, 3> A;
  std::array<std::array<T, 3>, 3> B;
  std::array<std::array<T, 3>, 3> C;

  // Set random values
  A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};
  B = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};
  C = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  tick("gemm");
  for (int i = 0; i < 100000000; ++i) {
    quad::gemm(A, B, C);
  }
  tock("gemm");

  quad::printMat("A", A);
  quad::printMat("B", B);
  quad::printMat("C", C);

  std::array<std::array<T, 3>, 3> D;
  quad::randFill(D);
  quad::printMat("D", D);

  quad::zeroMat(D);
  quad::printMat("D", D);

  quad::randFillSym(D);
  quad::printMat("D", D);

  T mu = 0.347;
  T lambda = 1.758;
  T wdetJ = 0.919;
  T dmu, dlambda;

  std::array<std::array<T, 3>, 3> Jinv, Uxi, Pxi;
  std::array<std::array<T, 3>, 3> Uxib;

  quad::randFill(Jinv);
  quad::randFill(Uxi);
  quad::randFill(Pxi);

  std::array<std::array<T, 3>, 3> Ux;
  quad::gemm(Uxi, Jinv, Ux);

  return 0;
}