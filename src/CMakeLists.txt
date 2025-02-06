if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
else()
    set(GGML_STANDALONE OFF)
endif()

set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)

add_subdirectory(${GGML_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED gtk+-3.0)

include_directories(${GTK_INCLUDE_DIRS})
link_directories(${GTK_LIBRARY_DIRS})
add_definitions(${GTK_CFLAGS_OTHER})

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

set(GUI_SOURCES gui.c)
set(GUI_TARGET sam_gui)

add_executable(${GUI_TARGET} ${GUI_SOURCES})
target_link_libraries(${GUI_TARGET} PRIVATE 
    ggml
    sam
    sam_c
    ${GTK_LIBRARIES}
)

# Copy SAM model file to binary directory
add_custom_command(TARGET sam_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/weights/sam_vit_b-ggml-model-f16.bin
        $<TARGET_FILE_DIR:sam_gui>/sam_vit_b-ggml-model-f16.bin)