if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
else()
    set(GGML_STANDALONE OFF)
endif()

set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)

add_subdirectory(${GGML_DIR})

add_library(sam STATIC
    sam.cpp
    sam.h
)
add_library(sam_c STATIC
    sam-c.cpp
    sam-c.h
)

set(CLI_SOURCES cli.c)
set(CLI_TARGET sam_cli)

add_executable(${CLI_TARGET} ${CLI_SOURCES})
target_link_libraries(${CLI_TARGET} PRIVATE 
    ggml
    sam
    sam_c
)
