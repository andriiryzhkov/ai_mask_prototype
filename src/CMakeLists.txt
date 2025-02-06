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

add_library(sam SHARED
    sam.cpp
    sam.h
)
add_library(sam_c SHARED
    sam-c.cpp
    sam-c.h
)

# Set version properties
set_target_properties(sam PROPERTIES
    VERSION 1.0.0
    SOVERSION 1)

set_target_properties(sam_c PROPERTIES 
    VERSION 1.0.0
    SOVERSION 1)

target_link_libraries(sam PRIVATE ggml)
target_link_libraries(sam_c PRIVATE sam)

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

include(GNUInstallDirs)

# Install executables
install(TARGETS ${CLI_TARGET} ${GUI_TARGET}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install model file
install(FILES ${CMAKE_SOURCE_DIR}/weights/sam_vit_b-ggml-model-f16.bin
    DESTINATION ${CMAKE_INSTALL_DATADIR}/sam)

# When running from installed location, look for model in installed path
target_compile_definitions(${CLI_TARGET} PRIVATE 
    SAM_MODEL_PATH="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/sam/sam_vit_b-ggml-model-f16.bin")
target_compile_definitions(${GUI_TARGET} PRIVATE 
    SAM_MODEL_PATH="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/sam/sam_vit_b-ggml-model-f16.bin")

# Find required DLLs 
if(WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK REQUIRED gtk+-3.0)

    execute_process(
        COMMAND pkg-config --variable=prefix gtk+-3.0
        OUTPUT_VARIABLE GTK_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Install libraries
    install(TARGETS sam sam_c ggml
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Install only required GTK DLLs
    set(REQUIRED_DLLS
        libgtk-3-0.dll
        libgdk-3-0.dll
        libgdk_pixbuf-2.0-0.dll
        libpango-1.0-0.dll
        libpangocairo-1.0-0.dll
        libpangoft2-1.0-0.dll
        libpangowin32-1.0-0.dll
        libglib-2.0-0.dll
        libgobject-2.0-0.dll
        libcairo-2.dll
        libfontconfig-1.dll
        libfreetype-6.dll
        libharfbuzz-0.dll
        zlib1.dll
    )

    foreach(DLL ${REQUIRED_DLLS})
        if(EXISTS "${GTK_PREFIX}/bin/${DLL}")
            install(FILES "${GTK_PREFIX}/bin/${DLL}" 
                DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    endforeach()

    # Install minimal required runtime data
    install(DIRECTORY 
        "${GTK_PREFIX}/lib/gdk-pixbuf-2.0"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING PATTERN "*.dll"
        PATTERN "*.cache"
    )
endif()

install(FILES 
    sam.h
    sam-c.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sam)