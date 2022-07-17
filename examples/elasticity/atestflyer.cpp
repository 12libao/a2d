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

#include "viewwrapper.h"

using namespace std;
using namespace A2DKokkos;
using namespace A2D;

int main() {
  Kokkos::initialize();
  {
    // Kokkos::print_configuration();

    /// somewhere before dev kernel
    std::vector<double> xx;
    // initialize vector on host, with data dat no longer changes
    for (int i = 0; i < pow(2, 24); ++i) {
      xx.push_back(1);
    }

    int n = pow(2, 9);
    int m = pow(2, 9);
    // create 2d array all elements are 1
    double arr2D[n][m] = {1};
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) arr2D[i][j] = 1;

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
    vector<vector<double>> vec(n, vector<double>(m, 1.0));
    // print vec size
    // printf("%d %d\n", vec.size(), vec[0].size());

    // create 3d array all elements are 1
    int a = pow(2, 6);
    int b = pow(2, 6);
    int c = pow(2, 6);
    double arr3D[a][b][c] = {1};
    for (int i = 0; i < a; i++)
      for (int j = 0; j < b; j++)
        for (int k = 0; k < c; k++) arr3D[i][j][k] = 1;

    int rows3D = sizeof(arr3D) / sizeof(arr3D[0]);
    int cols3D = sizeof(arr3D[0]) / sizeof(arr3D[0][0]);
    int depth3D = sizeof(arr3D[0][0]) / sizeof(arr3D[0][0][0]);

    Kokkos::Timer timer;

    A2DKokkos::viewWrapper<double> x_view0;
    A2DKokkos::hostView1D<double> h_data;
    A2DKokkos::devView1D<double> d_data;
    h_data = x_view0.host1D(&xx[0], xx.size());
    d_data = x_view0.dev1D(&xx[0], xx.size());

    // double time = timer.seconds();
    printf("time( %g s )\n", timer.seconds());
    timer.reset();

    A2DKokkos::viewWrapper<double> vec2d_view;
    A2DKokkos::hostView2D<double> h_data2;
    A2DKokkos::devView2D<double> d_data2;
    // h_data2 = vec2d_view.host2D(&arr2D[0][0], n, m);
    // d_data2 = vec2d_view.dev2D(&arr2D[0][0], n, m);

    h_data2 = vec2d_view.host2D(&vec[0][0], n, m);
    d_data2 = vec2d_view.dev2D(&vec[0][0], n, m);

    printf("time( %g s )\n", timer.seconds());
    timer.reset();

    A2DKokkos::viewWrapper<double> vec3d_view;
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
    Kokkos::View<double ***> h_data4(&arr3D2[0][0][0], a, b, c);
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

    // printf("%lf\n", h_data(0));
    // printf("%lf\n", h_data2(10, 10));
    // printf("%lf\n", h_data3(10, 10, 10));


    constexpr index_t M = 100;
    constexpr index_t N = 360;

    // initialize vector on host
    int ndim = 1;
    CLayout<M, N> layout_mn(ndim);
    MultiArray<double, CLayout<M, N>> A(layout_mn);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        A(0, i, j) = 1.0;
      }
    }

    CLayout<M> layout_m(ndim);
    MultiArray<double, CLayout<M>> y(layout_m);
    for (int i = 0; i < M; ++i) {
      y(0, i) = 1.0;
    }

    CLayout<N> layout_n(ndim);
    MultiArray<double, CLayout<N>> x(layout_n);
    for (int i = 0; i < N; ++i) {
      x(0, i) = 1.0;
    }
    // ================================================================

    // timer.reset();
    // for (int i = 0; i < M; ++i) {
    //   double prod = 0.0;
    //   for (int j = 0; j < N; ++j) {
    //     prod += A(0, i, j) * x(0, j);
    //   }
    //   y(0, i) = prod;
    // }
    // printf("a2d time( %g s )\n", timer.seconds());

    A2DKokkos::blockGemv<double, M, N>(A, x, y);
    printf("y = %lf\n", y(0, 0));

    // timer.reset();
    // Kokkos::View<double *> d_x("d", N);
    // auto xHost = Kokkos::create_mirror_view(d_x);
    // Kokkos::View<double *, Kokkos::HostSpace> h_x(&x(0, 0), N);
    // Kokkos::deep_copy(xHost, h_x);
    // Kokkos::deep_copy(d_x, h_x);
    // printf("gemv time( %g s )\n", timer.seconds());
    // printf("NO ERROR ABOVE\n");
    // printf("xHost(0) = %lf\n", xHost(0));

    // timer.reset();

    // Kokkos::View<double *> d_x2("d", N);
    // Kokkos::View<double *, Kokkos::HostSpace> h_x2(&x(0, 0), N);
    // Kokkos::deep_copy(d_x2, h_x2);
    // printf("gemv time( %g s )\n", timer.seconds());
    // printf("NO ERROR ABOVE\n");
    // // printf("xHost(0) = %lf\n", xHost2(0));

    // Kokkos::View<double *> d_y("d", M);
    // auto yHost = Kokkos::create_mirror_view(d_y);
    // Kokkos::View<double *, Kokkos::HostSpace> h_y(&y(0, 0), M);
    // Kokkos::deep_copy(yHost, h_y);
    // Kokkos::deep_copy(d_y, h_y);

    // timer.reset();

    // Kokkos::View<double **> d_A("d", M, N);
    // auto AHost = Kokkos::create_mirror_view(d_A);
    // Kokkos::View<double **, Kokkos::HostSpace> h_A(&A(0, 0, 0), M, N);
    // Kokkos::deep_copy(AHost, h_A);
    // Kokkos::deep_copy(d_A, AHost);
    // printf("gemv time( %g s )\n", timer.seconds());
    // printf("A NO ERROR ABOVE\n");

    // ================================================================

    // A2DKokkos::viewWrapper<double> view;

    // A2DKokkos::hostView1D<double> h_y = view.host1D(&y(0, 0), M);
    // A2DKokkos::hostView2D<double> h_A = view.host2D(&A(0, 0, 0), M, N);
    // A2DKokkos::devView1D<double> d_x = view.dev1D(&x(0, 0), N);
    // A2DKokkos::devView1D<double> d_y = view.dev1D(&y(0, 0), M);
    // A2DKokkos::devView2D<double> d_A = view.dev2D(&A(0, 0, 0), M, N);

    // A2DKokkos::devView1D<double> d_y = view.devEmpty1D(M);
    // A2DKokkos::devView2D<double> d_A = view.devEmpty2D(M, N);

    // auto yHost = Kokkos::create_mirror_view(d_y);
    // auto AHost = Kokkos::create_mirror_view(d_A);

    // Kokkos::deep_copy(yHost, h_y);
    // Kokkos::deep_copy(AHost, h_A);
    // printf("NO ERROR ABOVE\n");

    // Kokkos::deep_copy(d_x, xHost);
    // Kokkos::deep_copy(d_y, yHost);
    // Kokkos::deep_copy(d_A, AHost);

    // xHost = view.host1D(&x(0, 0), N);
    // yHost = view.host1D(&y(0, 0), M);
    // AHost = view.host2D(&A(0, 0, 0), M, N);

    // printf("cc dd NO ERROR ABOVE\n");

    // printf("%lf\n", h_A(4, 10));
    // printf("%lf\n", AHost(4, 10));
    // Kokkos::deep_copy(d_x, h_x);
    // Kokkos::deep_copy(d_y, h_y);
    // Kokkos::deep_copy(d_A, h_A);

    // KokkosBlas::gemv("N", 1.0, d_A, d_x, 0.0, d_y);
    // #define KOKKOS_INLINE_FUNCTION inline __device__ __host__;
    // timer.reset();
    // // // KOKKOS_INLINE_FUNCTION
    // KokkosBlas::gemv("N", 1.0, d_A, d_x, 0.0, d_y);
    // printf("gemv time( %g s )\n", timer.seconds());

    // Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> y_view(
    //     &y(0, 0), M);
    // Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> x_view(
    //     &x(0, 0), N);
    // Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> A_view(
    //     &A(0, 0, 0), M, N);

    // timer.reset();
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace>
    // y_view("y",
    //                                                                       M);
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
    // yhost_view(
    //     &y(0, 0), M);
    // Kokkos::deep_copy(y_view, yhost_view);
    // printf("no error above\n");
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace,
    //              Kokkos::MemoryUnmanaged>
    //     y_view(&y(0, 0), M);
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace,
    //              Kokkos::MemoryUnmanaged>
    //     x_view(&x(0, 0), N);
    // Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::CudaSpace,
    //              Kokkos::MemoryUnmanaged>
    //     A_view(&A(0, 0, 0), M, N);

    // ================================================================
    // timer.reset();
    // typedef Kokkos::View<double *> ViewVectorType;
    // typedef Kokkos::View<double **> ViewMatrixType;
    // ViewVectorType dd_y("y", M);
    // ViewVectorType dd_x("x", N);
    // ViewMatrixType dd_A("A", M, N);

    // printf("ViewVectorType time( %g s )\n", timer.seconds());
    // timer.reset();
    // // Create host mirrors of device views.
    // auto hh_y = Kokkos::create_mirror_view(dd_y);
    // auto hh_x = Kokkos::create_mirror_view(dd_x);
    // auto hh_A = Kokkos::create_mirror_view(dd_A);

    // printf("HostMirror time( %g s )\n", timer.seconds());
    // timer.reset();

    // printf("Populate data && copy time( %g s )\n", timer.seconds());
    // timer.reset();

    // Kokkos::View<double *, Kokkos::HostSpace> hhh_y(&y(0, 0), M);
    // Kokkos::View<double *, Kokkos::HostSpace> hhh_x(&x(0, 0), N);
    // Kokkos::View<double **, Kokkos::HostSpace> hhh_A(&A(0, 0, 0), M, N);
    // Kokkos::View<double **, Kokkos::CudaSpace> dddd_A(&A(0, 0, 0), M, N);
    // // hh_y(&y(0, 0), M);

    // printf("Populate data time( %g s )\n", timer.seconds());
    // timer.reset();

    // Kokkos::deep_copy(hh_y, hhh_y);
    // Kokkos::deep_copy(hh_x, hhh_x);
    // Kokkos::deep_copy(hh_A, hhh_A);

    // printf("deep_copy data time( %g s )\n", timer.seconds());
    // timer.reset();

    // Kokkos::deep_copy(dd_y, hh_y);
    // Kokkos::deep_copy(dd_x, hh_x);
    // Kokkos::deep_copy(dd_A, hh_A);

    // printf("deep_copy h=>d time( %g s )\n", timer.seconds());
    // timer.reset();

    // KokkosBlas::gemv("N", 1.0, dddd_A, dd_x, 0.0, dd_y);

    // printf("gemv time( %g s )\n", timer.seconds());
    // timer.reset();
    // printf("hhh_y = %lf\n", hhh_y(0));
    // printf("hh_y = %lf\n", hh_y(0));
    // printf("y = %lf\n", y(0, 0));
    // Kokkos::deep_copy(hh_y, dd_y);
    // printf("deep_copy d=>h time( %g s )\n", timer.seconds());
    // printf("hhh_y = %lf\n", hhh_y(0));
    // printf("hh_y = %lf\n", hh_y(0));
    // printf("y = %lf\n", y(0, 0));
    // timer.reset();

    // Kokkos::deep_copy(hh_y, dd_y);
    // printf("deep_copy h=>h time( %g s )\n", timer.seconds());
    // printf("hhh_y = %lf\n", hhh_y(0));
    // printf("hh_y = %lf\n", hh_y(0));
    // printf("y = %lf\n", y(0, 0));
    // timer.reset();

    // printf("hh no error above\n");

    // printf("hhh_y = %lf\n", hhh_y(0));

    // ================================================================
    for (int i = 0; i < N; ++i) {
      x(0, i) = 0.001;
    }
    timer.reset();
    printf("&A(0, 0)= %p\n &x(0) = %p\n y(0)= %p\n ", &A(0, 0, 0), &x(0, 0),
           &y(0, 0));
    Kokkos::View<double *, Kokkos::CudaHostPinnedSpace> ddd_x(&x(0, 0), N);
    Kokkos::View<double *, Kokkos::CudaHostPinnedSpace> ddd_y(&y(0, 0), M);
    Kokkos::View<double **, Kokkos::CudaHostPinnedSpace> ddd_A(&A(0, 0, 0), M,
                                                               N);
    printf("Populate data time( %g s )\n", timer.seconds());

    timer.reset();
    // KokkosBlas::gemv("N", 1.0, ddd_A, ddd_x, 0.0, ddd_y);
    printf("gemv gpu time( %g s )\n", timer.seconds());
    printf("y = %lf\n", y(0, 0));
    printf("ddd_y = %lf\n", ddd_y(0));
    // printf("hhh_y = %lf\n", hhh_y(0));

    // ================================================================

    // timer.reset();
    // Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> y_view(
    //     &y(0, 0), M);
    // Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> x_view(
    //     &x(0, 0), N);
    // Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> A_view(
    //     &A(0, 0, 0), M, N);

    // KokkosBlas::gemv("N", 1.0, A_view, x_view, 0.0, y_view);
    // printf("a2d gemv time( %g s )\n", timer.seconds());
    // printf("y_view = %lf\n", y_view(0));
    // printf("y = %lf\n", y(0, 0));

    // for (int i = 0; i < N; ++i) {
    //   x(0, i) = 0.1;
    // }

    // timer.reset();
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace> y_view_cuda(
    //     &y(0, 0), M);
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace> x_view_cuda(
    //     &x(0, 0), N);
    // Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::CudaSpace> A_view_cuda(
    //     &A(0, 0, 0), M, N);
    // auto y_view_cuda_mirror = Kokkos::create_mirror_view(y_view_cuda);
    // Kokkos::deep_copy(y_view_cuda_mirror, y_view_cuda);
    // printf("y_view_cuda = %lf\n", y_view_cuda_mirror(0));
    // KokkosBlas::gemv("N", 1.0, A_view_cuda, x_view_cuda, 0.0, y_view_cuda);

    

    // printf("cuda gemv time( %g s )\n", timer.seconds());
    // printf("y_view_cuda = %lf\n", y_view_cuda_mirror(0));
    // printf("y = %lf\n", y(0, 0));

    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace,
    //              Kokkos::MemoryUnmanaged>::HostMirror y_view_host =
    //     Kokkos::create_mirror_view(y_view);
    // printf("271 no error above\n");

    // Kokkos::deep_copy(y_view, y_view_host);

    // printf("no error above\n");
    // Kokkos::parallel_for(
    //     Kokkos::RangePolicy<int>(0, M), KOKKOS_LAMBDA(int i) {
    //       double prod = 0.0;
    //       for (int j = 0; j < N; j++) {
    //         prod += A_view(i, j) * x_view(j);
    //       }
    //       y_view(i) = prod;
    //     });

    // printf("gemv time( %g s )\n", timer.seconds());

    // auto yHost2 = Kokkos::create_mirror(y_view);
    // printf("yHost2(0) = %lf\n", yHost2(0));
    // Kokkos::deep_copy(yHost2, y_view);
    // printf("yHost2(0) = %lf\n", yHost2(0));
    // printf("y_view(0) = %lf\n", y_view(0));
    // Kokkos::deep_copy(yhost_view, y_view);
    // printf("no error above\n");
    // printf("yhost_view(0) = %lf\n", yhost_view(0));

    // A2DKokkos::blockGemv<double, M, N>(A_view, x_view, y_view);
    // printf("h_y = %lf\n", h_y(0));
    // printf("yHost = %lf\n", yHost(0));

    // Kokkos::deep_copy(y, y_view);
    // timer.reset();
    // // KOKKOS_INLINE_FUNCTION
    // KokkosBlas::gemv("N", 1.0, A_view, x_view, 0.0, d_y);
    // printf("gemv time( %g s )\n", timer.seconds());
    // Kokkos::deep_copy(h_y, d_y);
    // Kokkos::deep_copy(y, y_view);
    // printf("y(0) = %lf\n", h_y(0));
    // printf("y = %lf\n", y(1, 1));

    // for (int i = 0; i < M; ++i) {
    //   printf("%lf\n", y(0, i));
    // }

    // // auto y = static_cast<double *>(std::malloc(M * sizeof(double)));
    // // auto x = static_cast<double *>(std::malloc(N * sizeof(double)));
    // // auto A = static_cast<double *>(std::malloc(M * N * sizeof(double)));

    // // // Initialize y vector.
    // // Kokkos::parallel_for(
    // //     "y_init", N, KOKKOS_LAMBDA(int i) { y[i] = 1; });

    // // // Initialize x vector.
    // // Kokkos::parallel_for(
    // //     "x_init", M, KOKKOS_LAMBDA(int i) { x[i] = 1; });

    // // // Initialize A matrix, note 2D indexing computation.
    // // Kokkos::parallel_for(
    // //     "matrix_init", N, KOKKOS_LAMBDA(int j) {
    // //       for (int i = 0; i < M; ++i) {
    // //         A[j * M + i] = 1;
    // //       }
    // //     });

    // using Scalar = double;
    // using ExecSpace = Kokkos::Cuda;
    // using Layout = Kokkos::LayoutRight;
    // using Device = Kokkos::Device<ExecSpace, ExecSpace::memory_space>;

    // // Create a View containing a 2D matrix; allocate KokkosView with
    // template
    // // args of Scalar**, a layout, and
    // Kokkos::View<Scalar **, Layout, Device> A(
    //     Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), m, n);
    // // Create Views containing 1D matrix; allocate (without) matrix "x" of
    // size
    // // n
    // Kokkos::View<Scalar *, Device> x(
    //     Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), n);
    // // Create Views containing 1D matrix; allocate (without) matrix "y" of
    // size
    // // m
    // Kokkos::View<Scalar *, Device> y(
    //     Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), m);

    // // Declaring variable pool w/ a number seed;
    // // a parallel random number generator, so you
    // // won't get the same number with a given seed each time
    // Kokkos::Random_XorShift64_Pool<ExecSpace> pool(123);

    // // Fill 2D Matrix "A" and 1D matrix (i.e., a vector) "x" with random
    // values;
    // // Here, 10 is the max value of the random generator between 1 and 10
    // // (uniform )
    // Kokkos::fill_random(A, pool, 10.0);
    // Kokkos::fill_random(x, pool, 10.0);

    // // Do a warm-up run
    // KokkosBlas::gemv("N", 1.0, A, x, 0.0, y);
    // Kokkos::fence();

    // vector<double> y(M, 1.0);
    // vector<double> x(N, 1.0);
    // vector<vector<double>> A(M, vector<double>(N, 1.0));

    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace>
    // y_view(&y[0],
    //                                                                     M);
    // Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::CudaSpace>
    // x_view(&x[0],
    //                                                                     N);
    // Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::CudaSpace> A_view(
    //     &A[0][0], M, N);

    // ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view(y_view);
    // ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view(x_view);
    // ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view(add_view);

    // const double alpha = double(1.0);
    // const double beta = double(0.0);

    // KokkosBlas::gemv("N", alpha, A_view, x_view, beta, y_view);

    // Kokkos::deep_copy(h_y, y_view);
    // Kokkos::deep_copy(h_x, x_view);
    // Kokkos::deep_copy(h_A, A_view);

    // A2DKokkos::blockGemv<double, M, N>(A_view, x_view, y_view);
    // Print out the result.

    // std::free(A);
    // std::free(y);
    // std::free(x);
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
//     // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N )
//     ); double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

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

//   // If both N and M are undefined, fix row length to the smaller of S and
//   2
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