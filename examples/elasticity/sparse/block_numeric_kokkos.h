#ifndef BLOCK_NUMERIC_KOKKOS_H
#define BLOCK_NUMERIC_KOKKOS_H

#include <sys/time.h>

#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_team_dot.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>

#include "a2dtmp.h"
// #include <Kokkos_Timer.hpp>
// #include "block_numeric.h"
#include "block_numeric_kokkos.h"
#include "multiarray.h"
// #include "sparse_matrix.h"
// #include "sparse_numeric.h"
// #include "sparse_symbolic.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace A2DKokkos {

double fabs_a2d(std::complex<double> a) {
  if (a.real() >= 0.0) {
    return a.real();
  } else {
    return -a.real();
  }
}

double fabs_a2d(double a) {
  if (a >= 0.0) {
    return a;
  } else {
    return -a;
  }
}

double RealPart(double a) { return a; }

double RealPart(std::complex<double> a) { return a.real(); }

/*
  Compute y = A * x
*/
template <typename T, int M, int N, class AType, class xType, class yType>
KOKKOS_INLINE_FUNCTION void blockGemv(const AType& A, const xType& x, yType& y) {
  printf("============\n");
  printf("A(0, 0)= %f\n x(0) = %f\n y(0)= %f\n ", A(0, 0, 0), x(0, 0), y(0, 0));
  printf("&A(0, 0)= %p\n &x(0) = %p\n y(0)= %p\n ", &A(0, 0, 0), &x(0, 0),
         &y(0, 0));
  // Kokkos::View<T*, Kokkos::CudaHostPinnedSpace> ddd_x(&x(0, 0), N);
  // Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> d_y(&y(0, 0), M);
  // Kokkos::View<double**, Kokkos::CudaHostPinnedSpace> d_A(&A(0, 0, 0), M, N);
  // KokkosBlas::gemv("N", 1.0, d_A, d_x, 0.0, d_y);

  // ============================================================
  // Kokkos::parallel_for(
  //     Kokkos::RangePolicy<int>(0, M), KOKKOS_LAMBDA(int i) {
  //       T prod = 0.0;
  //       for (int j = 0; j < N; j++) {
  //         prod += A(i, j) * x(j);
  //       }
  //       y(i) = prod;
  //     });
  // ================================================================
  // for (int i = 0; i < M; i++) {
  //   T prod = 0.0;
  //   for (int j = 0; j < N; j++) {
  //     prod += A(i, j) * x(j);
  //   }
  //   y(i) = prod;
  // }
  printf("===========\n");
}

/*
  Compute y += A * x
*/
template <typename T, int M, int N, class AType, class xType, class yType>
inline void blockGemvAdd(const AType& A, const xType& x, yType& y) {
  for (int i = 0; i < M; i++) {
    T prod = 0.0;
    for (int j = 0; j < N; j++) {
      prod += A(i, j) * x(j);
    }
    y(i) += prod;
  }
}

/*
  Compute y -= A * x
*/
template <typename T, int M, int N, class AType, class xType, class yType>
inline void blockGemvSub(const AType& A, const xType& x, yType& y) {
  for (int i = 0; i < M; i++) {
    T prod = 0.0;
    for (int j = 0; j < N; j++) {
      prod += A(i, j) * x(j);
    }
    y(i) -= prod;
  }
}

/*
  Compute y = scale * A * x
*/
template <typename T, int M, int N, class AType, class xType, class yType>
inline void blockGemvScale(T scale, const AType& A, const xType& x, yType& y) {
  for (int i = 0; i < M; i++) {
    T prod = 0.0;
    for (int j = 0; j < N; j++) {
      prod += A(i, j) * x(j);
    }
    y(i) = scale * prod;
  }
}

/*
  Compute y += scale * A * x
*/
template <typename T, int M, int N, class AType, class xType, class yType>
inline void blockGemvAddScale(T scale, const AType& A, const xType& x,
                              yType& y) {
  for (int i = 0; i < M; i++) {
    T prod = 0.0;
    for (int j = 0; j < N; j++) {
      prod += A(i, j) * x(j);
    }
    y(i) += scale * prod;
  }
}

/*
  Compute: C = A * B

  A in M x N
  B in N x P
  C in M x P
*/
template <typename T, int M, int N, int P, class AType, class BType,
          class CType>
inline void blockGemm(const AType& A, const BType& B, CType& C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      T prod = 0.0;
      for (int k = 0; k < N; k++) {
        prod += A(i, k) * B(k, j);
      }
      C(i, j) = prod;
    }
  }
}

/*
  Compute: C += A * B

  A in M x N
  B in N x P
  C in M x P
*/
template <typename T, int M, int N, int P, class AType, class BType,
          class CType>
inline void blockGemmAdd(const AType& A, const BType& B, CType& C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      T prod = 0.0;
      for (int k = 0; k < N; k++) {
        prod += A(i, k) * B(k, j);
      }
      C(i, j) += prod;
    }
  }
}

/*
  Compute: C -= A * B

  A in M x N
  B in N x P
  C in M x P
*/
template <typename T, int M, int N, int P, class AType, class BType,
          class CType>
inline void blockGemmSub(const AType& A, const BType& B, CType& C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      T prod = 0.0;
      for (int k = 0; k < N; k++) {
        prod += A(i, k) * B(k, j);
      }
      C(i, j) -= prod;
    }
  }
}

/*
  Compute: C = scale * A * B

  A in M x N
  B in N x P
  C in M x P
*/
template <typename T, int M, int N, int P, class AType, class BType,
          class CType>
inline void blockGemmScale(T scale, const AType& A, const BType& B, CType& C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      T prod = 0.0;
      for (int k = 0; k < N; k++) {
        prod += A(i, k) * B(k, j);
      }
      C(i, j) = scale * prod;
    }
  }
}

/*
  Compute: C += scale * A * B

  A in M x N
  B in N x P
  C in M x P
*/
template <typename T, int M, int N, int P, class AType, class BType,
          class CType>
inline void blockGemmAddScale(T scale, const AType& A, const BType& B,
                              CType& C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      T prod = 0.0;
      for (int k = 0; k < N; k++) {
        prod += A(i, k) * B(k, j);
      }
      C(i, j) += scale * prod;
    }
  }
}

/*
  Compute: Ainv = A^{-1} with pivoting
*/
template <typename T, int N, class AType, class AinvType, class IType>
int blockInverse(AType& A, AinvType& Ainv, IType& ipiv) {
  int fail = 0;

  for (int k = 0; k < N - 1; k++) {
    // Find the maximum value and use it as the pivot
    int r = k;
    T maxv = A(k, k);
    for (int j = k + 1; j < N; j++) {
      T t = A(j, k);
      if (fabs_a2d(t) > fabs_a2d(maxv)) {
        maxv = t;
        r = j;
      }
    }

    ipiv(k) = r;

    // If a swap is required, swap the rows
    if (r != k) {
      for (int j = 0; j < N; j++) {
        T t = A(k, j);
        A(k, j) = A(r, j);
        A(r, j) = t;
      }
    }

    if (fabs_a2d(A(k, k)) == 0.0) {
      fail = k + 1;
      return fail;
    }

    for (int i = k + 1; i < N; i++) {
      A(i, k) = A(i, k) / A(k, k);
    }

    for (int i = k + 1; i < N; i++) {
      for (int j = k + 1; j < N; j++) {
        A(i, j) -= A(i, k) * A(k, j);
      }
    }
  }

  // Now, compute the matrix-inverse
  for (int k = 0; k < N; k++) {
    int ip = k;
    for (int i = 0; i < N - 1; i++) {
      if (ip == ipiv(i)) {
        ip = i;
      } else if (ip == i) {
        ip = ipiv(i);
      }
    }

    for (int i = 0; i < ip; i++) {
      Ainv(i, k) = 0.0;
    }

    Ainv(ip, k) = 1.0;

    for (int i = ip + 1; i < N; i++) {
      Ainv(i, k) = 0.0;
      for (int j = ip; j < i; j++) {
        Ainv(i, k) -= A(i, j) * Ainv(j, k);
      }
    }

    for (int i = N - 1; i >= 0; i--) {
      for (int j = i + 1; j < N; j++) {
        Ainv(i, k) -= A(i, j) * Ainv(j, k);
      }
      Ainv(i, k) = Ainv(i, k) / A(i, i);
    }
  }

  return fail;
}

}  // namespace A2DKokkos

#endif  // BLOCK_NUMERIC_H
