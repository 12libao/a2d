# include A2D headers
include_directories(${A2D_ROOT_DIR}/include)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX LAPACK::LAPACK)

add_executable(test_numeric test_numeric.cpp)
target_link_libraries(test_numeric PUBLIC OpenMP::OpenMP_CXX LAPACK::LAPACK)


# if enabled google test
option(GOOGLE_TEST "Enable Google Test" OFF)
if (GOOGLE_TEST)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.0
  )
  FetchContent_MakeAvailable(googletest)

  add_executable(runUnitTests test_quad.cpp)
  target_link_libraries(runUnitTests PUBLIC OpenMP::OpenMP_CXX LAPACK::LAPACK gtest_main)
endif()

# Add targets

# add_executable(elasticity elasticity.cpp)
# add_executable(test_grad_hess test_grad_hess.cpp)
# add_executable(test_numeric test_numeric.cpp)
# add_executable(toy_elasticity toy_elasticity.cpp)

# Link libraries

# target_link_libraries(elasticity     OpenMP::OpenMP_CXX LAPACK::LAPACK)
# target_link_libraries(test_grad_hess OpenMP::OpenMP_CXX LAPACK::LAPACK)
# target_link_libraries(test_numeric   OpenMP::OpenMP_CXX LAPACK::LAPACK)
# target_link_libraries(toy_elasticity OpenMP::OpenMP_CXX LAPACK::LAPACK)

# # Link to kokkos if used
# if (A2D_USE_KOKKOS)
#   # target_link_libraries(runUnitTests        kokkos)
#   # target_link_libraries(elasticity     kokkos)
#   # target_link_libraries(test_grad_hess kokkos)
#   # target_link_libraries(test_numeric   kokkos)
#   # target_link_libraries(toy_elasticity kokkos)
# endif()

