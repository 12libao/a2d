cmake_minimum_required(VERSION 3.22)

project(A2D)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3")

# add_subdirectory(a2d)
# add_subdirectory(sparse)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(KokkosKernels REQUIRED)

add_executable(atestflyer atestflyer.cpp)
target_link_libraries(atestflyer     
    PUBLIC
        Kokkos::kokkoskernels)

# target_include_directories(atestflyer PUBLIC a2d sparse)
# target_link_libraries(atestflyer     
#     PUBLIC
#         a2dlib
#         sparselib
#         Kokkos::kokkoskernels 
#         OpenMP::OpenMP_CXX 
#         MPI::MPI_CXX)

# add_executable(elasticity elasticity.cpp)
# target_include_directories(elasticity PUBLIC a2d sparse)
# target_link_libraries(elasticity 
#     PUBLIC
#         a2dlib
#         sparselib
#         Kokkos::kokkoskernels 
#         OpenMP::OpenMP_CXX 
#         MPI::MPI_CXX)

# add_executable(sparse_numerical sparse_numerical.cpp)
# target_link_libraries(sparse_numerical Kokkos::kokkos)
# target_link_libraries(sparse_numerical Kokkos::kokkoskernels)

# add_library(spnu sparse_numeric.h)
# target_sources(spnu PUBLIC sparse_numeric.h)
# target_compile_definitions(spnu PUBLIC -D KOKKOS_ENABLE_OPENMP -D KOKKOS_ENABLE_CUDA)
# target_link_libraries(spnu INTERFACE Kokkos::kokkos)
# target_compile_definitions(spnu PUBLIC -D KOKKOS_ENABLE_OPENMP)

# include_directories(${Kokkos_INCLUDE_PATH})
# find_path(SPNU_INCLUDE_DIR sparse_numeric.h)
# target_link_libraries(A2D Kokkos::kokkos)
# target_include_directories(A2D PUBLIC ${Kokkos_INCLUDE_DIRS} ${SPNU_INCLUDE_DIR})

# Build executables

# set_source_files_properties(sparse_numerical.cpp PROPERTIES LANGUAGE CUDA)

# target_compile_definitions(sparse_numeric PUBLIC -D KOKKOS_ENABLE_OPENMP -D KOKKOS_ENABLE_CUDA)




# add_executable(elasticity_kokkos elasticity_kokkos.cpp)
# target_link_libraries(elasticity_kokkos Kokkos::kokkos)
# target_include_directories(elasticity PUBLIC ${Kokkos_INCLUDE_DIRS})











# # Specify cmake version needed
# cmake_minimum_required(VERSION 3.22)

# # Set project name
# project(A2D VERSION 0.1)

# # Specify c++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Set compiler flags
# set(CMAKE_CXX_FLAGS "-g -O3")

# # Require MPI
# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})

# # Require OpenMP
# find_package(OpenMP REQUIRED)

# # Build executables
# add_executable(test_grad_hess test_grad_hess.cpp)
# target_link_libraries(test_grad_hess PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

# add_executable(elasticity elasticity.cpp)
# target_link_libraries(elasticity PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

# # Require pybind11 to build the extension
# execute_process(COMMAND python3 -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
# find_package(pybind11 REQUIRED)

# # Build extension
# pybind11_add_module(example example.cpp)

# # Download google test (git tag needs to be updated frequently)
# include(FetchContent)
# FetchContent_Declare(
#    googletest
#    GIT_REPOSITORY https://github.com/google/googletest.git
#    GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.10.0
# )
# FetchContent_MakeAvailable(googletest)

# # Declare tests
# add_subdirectory(tests)

# # Link to the default main from Google Test
# target_link_libraries(
#   test_matcore
#   gtest_main
# )

# target_link_libraries(
#   test_matcore2d
#   gtest_main
# )

# target_link_libraries(
#   test_tmp
#   gtest_main
# )

# target_link_libraries(
#   test_tmp2d
#   gtest_main
# )