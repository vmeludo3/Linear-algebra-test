add_library(benchmark_common INTERFACE)

target_include_directories(benchmark_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(benchmark_common INTERFACE
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${YAML_CPP_LIBRARY}
)

