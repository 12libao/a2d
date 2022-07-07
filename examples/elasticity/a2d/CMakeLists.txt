aux_source_directory(. DIR_LIB_SRCS)
add_library(a2dlib STATIC ${DIR_LIB_SRCS})
target_include_directories(a2dlib PUBLIC . ../sparse)