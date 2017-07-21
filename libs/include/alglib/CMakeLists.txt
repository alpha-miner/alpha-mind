file(GLOB_RECURSE ALGLIB_FILES ".h" "*.hpp" "*.cpp")

add_library(alglib ${LIB_TYPE} ${ALGLIB_FILES})
