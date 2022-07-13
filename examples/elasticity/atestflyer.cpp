#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
// #include <vector>

#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_team_dot.hpp>
#include <KokkosBlas2_gemv.hpp>
// #include <Kokkos_Timer.hpp>
#include "viewwrapper.h"

using namespace std;

int main() {
  Kokkos::initialize();
  {
    // Kokkos::print_configuration();

    /// somewhere before dev kernel
    std::vector<double> x;
    // initialize vector on host, with data dat no longer changes
    for (int i = 0; i < pow(2, 24); ++i) {
      x.push_back(1);
    }

    int n = pow(2, 9);
    int m = pow(2, 9);
    // create 2d array all elements are 1
    double arr2D[n][m] = {1};
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) 
        arr2D[i][j] = 1;

    // print m*n
    printf("%d\n", m * n);

    // int rows = sizeof(arr2D) / sizeof(arr2D[0]);
    // int cols = sizeof(arr2D[0]) / sizeof(arr2D[0][0]);
    // for (int i = 0; i < n; ++i) {
    //   for (int j = 0; j < m; ++j) {
    //     printf("%lf ", arr2D[i][j]);
    //   }
    //   printf("\n");
    // }
    // vector<vector<double>> vec(n, vector<double>(m, 1.0));
    // print vec size
    // printf("%d %d\n", vec.size(), vec[0].size());

    // create 3d array all elements are 1
    int a = pow(2, 6);
    int b = pow(2, 6);
    int c = pow(2, 6);
    double arr3D[a][b][c] = {1};
    for (int i = 0; i < a; i++)
      for (int j = 0; j < b; j++)
        for (int k = 0; k < c; k++)
          arr3D[i][j][k] = 1;

    int rows3D = sizeof(arr3D) / sizeof(arr3D[0]);
    int cols3D = sizeof(arr3D[0]) / sizeof(arr3D[0][0]);
    int depth3D = sizeof(arr3D[0][0]) / sizeof(arr3D[0][0][0]);

    Kokkos::Timer timer;

    A2DKokkos::viewWapper<double> x_view;
    A2DKokkos::hostView1D<double> h_data;
    A2DKokkos::devView1D<double> d_data;
    h_data = x_view.host1D(&x[0], x.size());
    d_data = x_view.dev1D(&x[0], x.size());

    // double time = timer.seconds();
    printf("time( %g s )\n", timer.seconds());
    timer.reset();

    A2DKokkos::viewWapper<double> vec2d_view;
    A2DKokkos::hostView2D<double> h_data2;
    A2DKokkos::devView2D<double> d_data2;
    h_data2 = vec2d_view.host2D(&arr2D[0][0], n, m);
    d_data2 = vec2d_view.dev2D(&arr2D[0][0], n, m);

    printf("time( %g s )\n", timer.seconds());
    timer.reset();

    A2DKokkos::viewWapper<double> vec3d_view;
    A2DKokkos::hostView3D<double> h_data3;
    A2DKokkos::devView3D<double> d_data3;
    h_data3 = vec3d_view.host3D(&arr3D[0][0][0], rows3D, cols3D, depth3D);
    d_data3 = vec3d_view.dev3D(&arr3D[0][0][0], rows3D, cols3D, depth3D);

    printf("time( %g s )\n", timer.seconds());
    
    a = pow(2, 6);
    b = pow(2, 6);
    c = pow(2, 6);
    double arr3D2[a][b][c] = {1};
    for (int i = 0; i < a; i++)
      for (int j = 0; j < b; j++)
        for (int k = 0; k < c; k++) arr3D2[i][j][k] = 1;
    timer.reset();
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::CudaSpace> h_data4(
        &arr3D2[0][0][0], a, b, c);
    printf("time( %g s )\n", timer.seconds());

    // h_data2 = vec2d_view.host2D(&vec[0][0], vec.size() , vec[0].size());
    // vec2d_view.dev2D(&vec[0][0], vec.size(), vec[0].size());

    // wrap with Kokkos View on Host
    // Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace,
    //              Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    //     h_data(&x[0], x.size());
    // Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
    //     dev_x(Kokkos::ViewAllocateWithoutInitializing("dev_x"),
    //     x.size());
    // Kokkos::deep_copy(dev_x, h_data);

    // print out the result
    // for (int i = 0; i < 100000000; ++i) {
    //   printf("%lf\n", dev_x(i));
    // }

    printf("%lf\n", h_data(0));
    printf("%lf\n", h_data2(10, 10));
    printf("%lf\n", h_data3(10, 10, 10));
    // for (int i = 0; i < n; ++i) {
    //   for (int j = 0; j < m; ++j) {
    //     printf("%lf ", h_data2(i, j));
    //   }
    //   printf("\n");
    // }

    // return time;

    // Use view in dev kernel
  }
  Kokkos::finalize();
  return 0;
}

// Use view in dev kernel

// int main() {
//   BSRMat<int, double, 3, 3> A(3, 3, 9, {0, 3, 6, 9},
//                               {0, 1, 2, 0, 1, 2, 0, 1, 2});

//   // print the matrix
//   for (int i = 0; i < A.num_rows(); i++) {
//     for (int j = 0; j < A.num_cols(); j++) {
//       printf("%f ", A(i, j));
//     }
//     printf("\n");
//   }
// }

// #include "block_numeric.h"
// #include "multiarray.h"
// #include "sparse_matrix.h"
// #include "sparse_symbolic.h"

// void checkSizes(int &N, int &M, int &S, int &nrepeat);

// int main(int argc, char *argv[]) {
//   int N = -1;         // number of rows 2^12
//   int M = -1;         // number of columns 2^10
//   int S = -1;         // total size 2^22
//   int nrepeat = 100;  // number of repeats of the test

//   // Read command line arguments.
//   for (int i = 0; i < argc; i++) {
//     if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0)) {
//       N = pow(2, atoi(argv[++i]));
//       printf("  User N is %d\n", N);
//     } else if ((strcmp(argv[i], "-M") == 0) ||
//                (strcmp(argv[i], "-Columns") == 0)) {
//       M = pow(2, atof(argv[++i]));
//       printf("  User M is %d\n", M);
//     } else if ((strcmp(argv[i], "-S") == 0) ||
//                (strcmp(argv[i], "-Size") == 0)) {
//       S = pow(2, atof(argv[++i]));
//       printf("  User S is %d\n", S);
//     } else if (strcmp(argv[i], "-nrepeat") == 0) {
//       nrepeat = atoi(argv[++i]);
//     }
//   }

//   // Check sizes.
//   checkSizes(N, M, S, nrepeat);

//   Kokkos::initialize(argc, argv);
//   {
// #ifdef KOKKOS_ENABLE_CUDA
// #define MemSpace Kokkos::CudaSpace
// #endif

//     // #ifndef KOKKOS_ENABLE_OPENMP
//     // #define MemSpace Kokkos::HostSpace
//     // #endif

//     // typedef Kokkos::DefaultExecutionSpace::array_layout  Layout;
//     // typedef Kokkos::LayoutLeft   Layout;
//     typedef Kokkos::LayoutRight Layout;

//     // using ExecSpace = MemSpace::execution_space;
//     // using range_policy = Kokkos::RangePolicy<ExecSpace>;

//     // Allocate y, x vectors and Matrix A on dev.
//     typedef Kokkos::View<double *, Layout, MemSpace> ViewVectorType;
//     typedef Kokkos::View<double **, Layout, MemSpace> 2DType;
//     ViewVectorType y("y", N);
//     ViewVectorType x("x", M);
//     2DType A("A", N, M);

//     // Create host mirrors of dev views.
//     ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view(y);
//     ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view(x);
//     2DType::HostMirror h_A = Kokkos::create_mirror_view(A);

//     // Initialize y vector on host.
//     for (int i = 0; i < N; ++i) {
//       h_y(i) = 1;
//     }

//     // Initialize x vector on host.
//     for (int i = 0; i < M; ++i) {
//       h_x(i) = 1;
//     }

//     // Initialize A matrix on host.
//     for (int j = 0; j < N; ++j) {
//       for (int i = 0; i < M; ++i) {
//         h_A(j, i) = 1;
//       }
//     }

//     // Deep copy host views to dev views.
//     Kokkos::deep_copy(y, h_y);
//     Kokkos::deep_copy(x, h_x);
//     Kokkos::deep_copy(A, h_A);

//     typedef Kokkos::TeamPolicy<> team_policy;
//     typedef Kokkos::TeamPolicy<>::member_type member_type;

//     // Timer products.
//     struct timeval begin, end;

//     //--------------------------------------------------------//
//     //------------------       Ex. 1        ------------------//
//     //------------------Using BLAS functions------------------//
//     //--------------------------------------------------------//

//     printf("  Using BLAS functions: gemv, dot\n");

//     ViewVectorType tmp("tmp", N);
//     double alpha = 1;
//     double beta = 0;

//     // gettimeofday(&begin, NULL);
//     Kokkos::Timer timer;
//     for (int repeat = 0; repeat < nrepeat; repeat++) {
//       // Application: <y,Ax> = y^T*A*x
//       double result = 0;
//       KokkosBlas::gemv("N", alpha, A, x, beta, tmp);
//       result = KokkosBlas::dot(y, tmp);

//       // Output result.
//       if (repeat == (nrepeat - 1)) {
//         printf("    Computed result for %d x %d is %lf\n", N, M, result);
//       }

//       const double solution = (double)N * (double)M;

//       if (result != solution) {
//         printf("    Error: result( %lf ) != solution( %lf )\n", result,
//                solution);
//       }
//     }

//     // gettimeofday(&end, NULL);
//     double time = timer.seconds();
//     // Calculate bandwidth.
//     // Each matrix A row (each of length M) is read once.
//     // The x vector (of length M) is read N times.
//     // The y vector (of length N) is read once.
//     // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
//     double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

//     // Print results (problem size, time and bandwidth in GB/s).
//     printf(
//         "    N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) "
//         "bandwidth( %g GB/s )\n",
//         N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
//   }

//   Kokkos::finalize();

//   return 0;
// }

// void checkSizes(int &N, int &M, int &S, int &nrepeat) {
//   // If S is undefined and N or M is undefined, set S to 2^22 or the bigger
//   of N
//       // and M.
//       if (S == -1 && (N == -1 || M == -1)) {
//     S = pow(2, 22);
//     if (S < N) S = N;
//     if (S < M) S = M;
//   }

//   // If S is undefined and both N and M are defined, set S = N * M.
//   if (S == -1) S = N * M;

//   // If both N and M are undefined, fix row length to the smaller of S and 2
//   ^
//       // 10 = 1024.
//       if (N == -1 && M == -1) {
//     if (S > 1024) {
//       N = pow(2, 15);
//     } else {
//       M = S;
//     }
//   }

//   // If only M is undefined, set it.
//   if (M == -1) M = S / N;

//   // If N is undefined, set it.
//   if (N == -1) N = S / M;

//   printf("  Total size S = %d N = %d M = %d\n", S, N, M);

//   // Check sizes.
//   if ((S < 0) || (N < 0) || (M < 0) || (nrepeat < 0)) {
//     printf("  Sizes must be greater than 0.\n");
//     exit(1);
//   }

//   if ((N * M) != S) {
//     printf("  N * M != S\n");
//     exit(1);
//   }
// }