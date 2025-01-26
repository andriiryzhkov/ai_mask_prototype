if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
else()
    set(GGML_STANDALONE OFF)
endif()

set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)

add_subdirectory(${GGML_DIR})

set(SOURCES main.cpp)

add_executable(sam ${SOURCES})

target_include_directories(sam PRIVATE ${GGML_DIR}/include)
target_link_libraries(sam PRIVATE ggml)
