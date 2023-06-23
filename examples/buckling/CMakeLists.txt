# include A2D headers
include_directories(${A2D_ROOT_DIR}/include)

# Set up paropt library and dependencies
find_package(MPI REQUIRED)
link_directories($ENV{HOME}/git/paropt/lib)
include_directories(${MPI_INCLUDE_PATH})
include_directories($ENV{HOME}/git/paropt/src)

# Add targets
add_executable(toy_elasticity toy_elasticity.cpp)
add_executable(toy_grad_check toy_grad_check.cpp)

# Link to kokkos, note that linking to kokkos must happen before
# liking to OpenMP::OpenMP, otherwise it might cause compile error
target_link_libraries(toy_elasticity Kokkos::kokkos)
target_link_libraries(toy_grad_check Kokkos::kokkos)

# Link libraries
target_link_libraries(toy_elasticity OpenMP::OpenMP_CXX LAPACK::LAPACK)
target_link_libraries(toy_grad_check OpenMP::OpenMP_CXX LAPACK::LAPACK)

# If using gcc and version < 9, need to explicitly link to filesystem
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        message("Using GCC ${CMAKE_CXX_COMPILER_VERSION} < 9.0.0, explicitly link to stdc++fs")
        target_link_libraries(toy_elasticity stdc++fs)
        target_link_libraries(toy_grad_check stdc++fs)
    endif()
endif()