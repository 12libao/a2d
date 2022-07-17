#ifndef VIEWWARP_H
#define VIEWWARP_H

/*
//@HEADER
// ************************************************************************
// This class is for wrapping existing pointers to Kokkos::Views
// View => pointer to raw memory
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

#include "KokkosKernels_default_types.hpp"

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

using Layout = default_layout;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using Device = Kokkos::Device<ExecSpace, ExecSpace::memory_space>;

template <typename T>
using hostView = Kokkos::View<T, Layout, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView1D = Kokkos::View<T *, Layout, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView2D = Kokkos::View<T **, Layout, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using hostView3D = Kokkos::View<T ***, Layout, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

#ifdef KOKKOS_ENABLE_CUDA  // dev view is a managed CUDA-view

template <typename T>
using devView =
    Kokkos::View<T, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using devView1D = Kokkos::View<T *, Layout, Device>;

template <typename T>
using devView2D = Kokkos::View<T **, Layout, Device>;

template <typename T>
using devView3D = Kokkos::View<T ***, Layout, Device>;

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
// A2DKokkos::viewWrapper<double> view2D;
// A2DKokkos::hostView2D<double> h_view2D;
// A2DKokkos::devView2D<double> d_view2D;
// h_view2D = view2D.host2D(&arr2D[0][0], rows, cols);
// d_view2D = view2D.dev2D(&arr2D[0][0], rows, cols);
// ************************************************************************
template <typename T>
class viewWrapper {
 public:
  A2DKokkos::hostView1D<T> host1D(T *buffer, int N0);
  A2DKokkos::hostView2D<T> host2D(T *buffer, int N0, int N1);
  A2DKokkos::hostView3D<T> host3D(T *buffer, int N0, int N1, int N2);
  A2DKokkos::devView1D<T> dev1D(T *buffer, int N0);
  A2DKokkos::devView2D<T> dev2D(T *buffer, int N0, int N1);
  A2DKokkos::devView3D<T> dev3D(T *buffer, int N0, int N1, int N2);
  A2DKokkos::devView1D<T> devEmpty1D(int N0);
  A2DKokkos::devView2D<T> devEmpty2D(int N0, int N1);
  A2DKokkos::devView3D<T> devEmpty3D(int N0, int N1, int N2);
  A2DKokkos::devView1D<T> popH2D1D(T *buffer, int N0);
  A2DKokkos::devView2D<T> popH2D2D(T *buffer, int N0, int N1);
  A2DKokkos::devView3D<T> popH2D3D(T *buffer, int N0, int N1, int N2);

  void syncH2D() { Kokkos::deep_copy(devview1D, hostview1D); }
  void syncD2H() { Kokkos::deep_copy(hostview1D, devview1D); }

  ~viewWrapper() {}

 private:
  A2DKokkos::devView1D<T> devview1D;
  A2DKokkos::devView2D<T> devview2D;
  A2DKokkos::devView3D<T> devview3D;
  A2DKokkos::hostView1D<T> hostview1D;
  A2DKokkos::hostView2D<T> hostview2D;
  A2DKokkos::hostView3D<T> hostview3D;
  A2DKokkos::devView1D<T> devviewempty1D;
  A2DKokkos::devView2D<T> devviewempty2D;
  A2DKokkos::devView3D<T> devviewempty3D;
};

// create a 1D view vector on the host
template <typename T>
A2DKokkos::hostView1D<T> viewWrapper<T>::host1D(T *buffer, int N0) {
  A2DKokkos::hostView1D<T> hostview1D(buffer, N0);
  return hostview1D;
}

// create a 2D view matrix on the host
template <typename T>
A2DKokkos::hostView2D<T> viewWrapper<T>::host2D(T *buffer, int N0, int N1) {
  A2DKokkos::hostView2D<T> hostview2D(buffer, N0, N1);
  return hostview2D;
}

// create a 3D view matrix on the host
template <typename T>
A2DKokkos::hostView3D<T> viewWrapper<T>::host3D(T *buffer, int N0, int N1,
                                                int N2) {
  A2DKokkos::hostView3D<T> hostview3D(buffer, N0, N1, N2);
  return hostview3D;
}

// create a 1D view vector on the device
template <typename T>
A2DKokkos::devView1D<T> viewWrapper<T>::dev1D(T *buffer, int N0) {
  A2DKokkos::devView1D<T> devview1D(buffer, N0);
  return devview1D;
}

// create a 2D view matrix on the device
template <typename T>
A2DKokkos::devView2D<T> viewWrapper<T>::dev2D(T *buffer, int N0, int N1) {
  A2DKokkos::devView2D<T> devview2D(buffer, N0, N1);
  return devview2D;
}

// create a 3D view matrix on the device
template <typename T>
A2DKokkos::devView3D<T> viewWrapper<T>::dev3D(T *buffer, int N0, int N1,
                                              int N2) {
  A2DKokkos::devView3D<T> devview3D(buffer, N0, N1, N2);
  return devview3D;
}

template <typename T>
A2DKokkos::devView1D<T> viewWrapper<T>::devEmpty1D(int N0) {
  A2DKokkos::devView1D<T> devviewempty1D(
      Kokkos::ViewAllocateWithoutInitializing("1D"), N0);
  return devview1D;
}

template <typename T>
A2DKokkos::devView2D<T> viewWrapper<T>::devEmpty2D(int N0, int N1) {
  A2DKokkos::devView2D<T> devviewempty2D(
      Kokkos::ViewAllocateWithoutInitializing("2D"), N0, N1);
  return devview2D;
}

template <typename T>
A2DKokkos::devView3D<T> viewWrapper<T>::devEmpty3D(int N0, int N1, int N2) {
  A2DKokkos::devView3D<T> devviewempty3D(
      Kokkos::ViewAllocateWithoutInitializing("3D"), N0, N1, N2);
  return devview3D;
}

template <typename T>
A2DKokkos::devView1D<T> viewWrapper<T>::popH2D1D(T *buffer, int N0) {
  // Allocate 1D on device.
  A2DKokkos::devView1D<T> devview1D("1D", N0);
  // Create host mirrors of device views.
  A2DKokkos::hostMirror1D<T> h_1D = Kokkos::create_mirror_view(devview1D);
  // Populate host mirrors on the host (from file, etc.)
  A2DKokkos::hostView1D<T> hostview1D(buffer, N0);
  Kokkos::deep_copy(h_1D, hostview1D);
  // Copy host mirrors to device views.
  Kokkos::deep_copy(devview1D, h_1D);
  return devview1D;
}

template <typename T>
A2DKokkos::devView2D<T> viewWrapper<T>::popH2D2D(T *buffer, int N0, int N1) {
  // Allocate 2D on device.
  A2DKokkos::devView2D<T> devview2D("2D", N0, N1);
  // Create host mirrors of device views.
  A2DKokkos::hostMirror2D<T> h_2D = Kokkos::create_mirror_view(devview2D);
  // Populate host mirrors on the host (from file, etc.)
  A2DKokkos::hostView2D<T> hostview2D(buffer, N0, N1);
  Kokkos::deep_copy(h_2D, hostview2D);
  // Copy host mirrors to device views.
  Kokkos::deep_copy(devview2D, h_2D);
  return devview2D;
}

template <typename T>
A2DKokkos::devView3D<T> viewWrapper<T>::popH2D3D(T *buffer, int N0, int N1,
                                                  int N2) {
  // Allocate 3D on device.
  A2DKokkos::devView3D<T> devview3D("3D", N0, N1, N2);
  // Create host mirrors of device views.
  A2DKokkos::hostMirror3D<T> h_3D = Kokkos::create_mirror_view(devview3D);
  // Populate host mirrors on the host (from file, etc.)
  A2DKokkos::hostView3D<T> hostview3D(buffer, N0, N1, N2);
  Kokkos::deep_copy(h_3D, hostview3D);
  // Copy host mirrors to device views.
  Kokkos::deep_copy(devview3D, h_3D);
  return devview3D;
}

}  // namespace A2DKokkos

#endif  // VIEWWARP_H
