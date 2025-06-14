# mtmd

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            mtmd-helper.cpp
            mtmd-helper.h
            )

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable         (${TARGET} mtmd-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
install                (TARGETS ${TARGET} RUNTIME)
target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET gex)
add_library         (${TARGET} SHARED gex.cpp gex.h)
target_compile_definitions(${TARGET} PRIVATE GEX_BUILD_SHARED)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME libgex)
install                (TARGETS ${TARGET} RUNTIME)
target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(GEX_PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/gex.h
)

set_target_properties(gex
        PROPERTIES
        PUBLIC_HEADER "${GEX_PUBLIC_HEADERS}")

install(TARGETS gex LIBRARY RUNTIME PUBLIC_HEADER)

add_executable(test_gex test_gex.cpp)
target_link_libraries(test_gex PRIVATE gex)