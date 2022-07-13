#ifndef VIEWWARP_H
#define VIEWWARP_H

/*
//@HEADER
// ************************************************************************
// This class is for wrapping existing pointers to Kokkos::Views
// View ? pointer to raw memory
// ************************************************************************
// Wrapping the memory in a View is a way to make it accessible to Kokkos.
// The View is a wrapper around the raw memory, and the raw memory is the
// underlying memory that is being accessed.The advantage is that the
// user does not need to know how the data is stored under the hood,

// Must specify everything about the View:
// View<ArraySpec, Layout, dev, Unmanaged> a(pointer, N0, N1, ...);
// Unmanaged views: allocates an array the regular way and defines a view
// on top of that, Kokkos cannot manage the memory
// dev: the memory must be on this dev
// { ArraySpec , Layout , N0 , N1 , ... }: the memory must have this shape
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

namespace A2DKokkos {

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#else
#define MemSpace Kokkos::HostSpace
#endif

// ************************************************************************
// If no layout specified, default for that memory space is used.
// LayoutLeft for CudaSpace, LayoutRight for HostSpace.
// LayoutRight <=>  C ordering <=> CLayout.
// ************************************************************************

template <typename T>
using hostView = Kokkos::View<T, Kokkos::LayoutRight, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView1D = Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView2D = Kokkos::View<T **, Kokkos::LayoutRight, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView3D = Kokkos::View<T ***, Kokkos::LayoutRight, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

#ifdef KOKKOS_ENABLE_CUDA  // dev view is a managed CUDA-view

template <typename T>
using devView = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::CudaSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using devView1D = Kokkos::View<T *, Kokkos::LayoutLeft, Kokkos::CudaSpace>;

template <typename T>
using devView2D = Kokkos::View<T **, Kokkos::LayoutLeft, Kokkos::CudaSpace>;

template <typename T>
using devView3D = Kokkos::View<T ***, Kokkos::LayoutLeft, Kokkos::CudaSpace>;

#else  // dev view is a managed host-view

template <typename T>
using devView = Kokkos::View<T, Kokkos::LayoutRight, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using devView1D = Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::HostSpace>;

template <typename T>
using devView2D = Kokkos::View<T **, Kokkos::LayoutRight, Kokkos::HostSpace>;

template <typename T>
using devView3D = Kokkos::View<T ***, Kokkos::LayoutRight, Kokkos::HostSpace>;

#endif

template <typename T>
using hostMirror = typename A2DKokkos::devView<T>::HostMirror;

template <typename T>
using hostMirror1D = typename A2DKokkos::devView1D<T>::HostMirror;

template <typename T>
using hostMirror2D = typename A2DKokkos::devView2D<T>::HostMirror;

template <typename T>
using hostMirror3D = typename A2DKokkos::devView3D<T>::HostMirror;

// ************************************************************************
// Class for wrapping kokkos views around existing pointers
// ************************************************************************
// This is how you would wrap a pointer to a C-style array in a Kokkos view.
// A2DKokkos::viewWapper<double> view2D;
// A2DKokkos::hostView2D<double> h_view2D;
// A2DKokkos::devView2D<double> d_view2D;
// h_view2D = view2D.host2D(&arr2D[0][0], rows, cols);
// d_view2D = view2D.dev2D(&arr2D[0][0], rows, cols);
// ************************************************************************
template <typename T>
class viewWapper {
 public:
  A2DKokkos::hostView1D<T> host1D(T *buffer, int N0);
  A2DKokkos::hostView2D<T> host2D(T *buffer, int N0, int N1);
  A2DKokkos::hostView3D<T> host3D(T *buffer, int N0, int N1, int N2);
  A2DKokkos::devView1D<T> dev1D(T *buffer, int N0);
  A2DKokkos::devView2D<T> dev2D(T *buffer, int N0, int N1);
  A2DKokkos::devView3D<T> dev3D(T *buffer, int N0, int N1, int N2);

  void syncH2D() { Kokkos::deep_copy(devview1D, hostview1D); }
  void syncD2H() { Kokkos::deep_copy(hostview1D, devview1D); }

  ~viewWapper() {}

 private:
  A2DKokkos::devView1D<T> devview1D;
  A2DKokkos::devView2D<T> devview2D;
  A2DKokkos::devView3D<T> devview3D;
  A2DKokkos::hostView1D<T> hostview1D;
  A2DKokkos::hostView2D<T> hostview2D;
  A2DKokkos::hostView3D<T> hostview3D;
};

// create a 1D view vector on the host
template <typename T>
A2DKokkos::hostView1D<T> viewWapper<T>::host1D(T *buffer, int N0) {
  A2DKokkos::hostView1D<T> hostview1D(buffer, N0);
  return hostview1D;
}

// create a 1D view vector on the device
template <typename T>
A2DKokkos::devView1D<T> viewWapper<T>::dev1D(T *buffer, int N0) {
  A2DKokkos::devView1D<T> devview1D(buffer, N0);
  return devview1D;
}

// create a 2D view matrix on the host
template <typename T>
A2DKokkos::hostView2D<T> viewWapper<T>::host2D(T *buffer, int N0, int N1) {
  A2DKokkos::hostView2D<T> hostview2D(buffer, N0, N1);
  return hostview2D;
}

// create a 2D view matrix on the device
template <typename T>
A2DKokkos::devView2D<T> viewWapper<T>::dev2D(T *buffer, int N0, int N1) {
  A2DKokkos::devView2D<T> devview2D(buffer, N0, N1);
  return devview2D;
}

// create a 3D view matrix on the host
template <typename T>
A2DKokkos::hostView3D<T> viewWapper<T>::host3D(T *buffer, int N0, int N1,
                                               int N2) {
  A2DKokkos::hostView3D<T> hostview3D(buffer, N0, N1, N2);
  return hostview3D;
}

// create a 3D view matrix on the device
template <typename T>
A2DKokkos::devView3D<T> viewWapper<T>::dev3D(T *buffer, int N0, int N1,
                                             int N2) {
  A2DKokkos::devView3D<T> devview3D(buffer, N0, N1, N2);
  return devview3D;
}

}  // namespace A2DKokkos

#endif  // VIEWWARP_H
