aux_source_directory(. DIR_LIB_SRCS)
add_library(sparselib STATIC ${DIR_LIB_SRCS})
target_include_directories(sparselib PUBLIC . ../a2d)