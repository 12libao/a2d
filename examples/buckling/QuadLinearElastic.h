#ifndef QUAD_LINEAR_ELASTIC_H
#define QUAD_LINEAR_ELASTIC_H

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>

namespace quad {

/*
 * Compute matrix multiplication C = A * B for 3x3 matrices
 *
 * Input:
 *   T A: 3x3 matrix  - row major, std::array<std::array<T, 3>, 3>
 *   T B: 3x3 matrix  - row major, std::array<std::array<T, 3>, 3>
 *   T C: 3x3 matrix  - row major, std::array<std::array<T, 3>, 3>
 *
 * Output:
 *   T C: 3x3 matrix  - row major, std::array<std::array<T, 3>, 3>
 *
 * Note:
 *   the data structure is std::array<std::array<T, 3>, 3>
 *   the data is stored in stack for better performance in flat memory
 */
template <typename T>
inline void gemm(const std::array<std::array<T, 3>, 3>& A,
                 const std::array<std::array<T, 3>, 3>& B,
                 std::array<std::array<T, 3>, 3>& C) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i][j] = 0.0;
      for (int k = 0; k < 3; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// template <typename T>
// class Strain





template <typename T>
void printMat(const char* name, std::array<std::array<T, 3>, 3>& A) {
  printf("Matrix: \033[32m%s\033[0m\n", name);
  for (int i = 0; i < 3; ++i) {
    printf("  |");
    for (int j = 0; j < 3; ++j) {
      printf("%9.5f ", A[i][j]);
    }
    printf("|\n");
  }
  printf("\n");
}

/*
 * Fill a 3x3 matrix with random values between min and max
 *
 * Input:
 *   T A: 3x3 matrix       - row major, std::array<std::array<T, 3>, 3>
 *   T min: minimum value  - default 0.0
 *   T max: maximum value  - default 1.0
 */
template <typename T>
void randFill(std::array<std::array<T, 3>, 3>& A, T min = -1.0, T max = 1.0) {
  static std::random_device rd;   // only need to initialize it once
  static std::mt19937 mte(rd());  // this is a relative big object to create

  std::uniform_real_distribution<T> dist(min, max);

  for (int i = 0; i < 3; ++i) {
    std::generate(A[i].begin(), A[i].end(), [&]() { return dist(mte); });
  }
}

/*
 * Fill a 3x3 matrix with random values between min and max
 * and make it symmetric
 */
template <typename T>
void randFillSym(std::array<std::array<T, 3>, 3>& A, T min = -1.0,
                 T max = 1.0) {
  randFill(A, min, max);
  A[1][0] = A[0][1];
  A[2][0] = A[0][2];
  A[2][1] = A[1][2];
}

// initialize a 3x3 matrix with zeros values
template <typename T>
inline void zeroMat(std::array<std::array<T, 3>, 3>& A) {
  std::fill(A.begin(), A.end(), std::array<T, 3>{0.0, 0.0, 0.0});
}

}  // namespace quad

#endif