if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
else()
    set(GGML_STANDALONE OFF)
endif()

set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)

add_subdirectory(${GGML_DIR})

add_library(sam STATIC sam.cpp sam.h)

set(CLI_SOURCES cli.cpp)
set(CLI_TARGET sam_cli)

add_executable(${CLI_TARGET} ${CLI_SOURCES})
target_include_directories(${CLI_TARGET} PRIVATE ${GGML_DIR}/include)
target_link_libraries(${CLI_TARGET} PRIVATE ggml sam)

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GTK REQUIRED gtk+-3.0)

# set(GUI_SOURCES main.cpp)
# set(GUI_TARGET sam_gui)

# add_executable(${GUI_TARGET} ${GUI_SOURCES})
# target_include_directories(${GUI_TARGET} PRIVATE ${GTK_INCLUDE_DIRS})
# target_link_directories(${GUI_TARGET} PRIVATE ${GTK_LIBRARY_DIRS})
# target_link_libraries(${GUI_TARGET} PRIVATE ${GTK_LIBRARIES} ggml sam)
# target_compile_options(${GUI_TARGET} PRIVATE ${GTK_CFLAGS_OTHER})