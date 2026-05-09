cmake_minimum_required(VERSION 3.18)
project(game_ggml
    VERSION 0.1.0
    DESCRIPTION "GAME (Generative Adaptive MIDI Extractor) native ggml inference backend"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(GAME_GGML_CUDA        "Enable ggml CUDA backend"     OFF)
option(GAME_GGML_VULKAN      "Enable ggml Vulkan backend"   OFF)
option(GAME_GGML_BUILD_CLI   "Build game_ggml_cli executable"   ON)
option(GAME_GGML_BUILD_TESTS "Build unit tests (requires GoogleTest via FetchContent)" OFF)
option(GAME_GGML_BUILD_WASM  "Build the WASM module (requires emcmake)" OFF)
option(GAME_GGML_WASM_THREADS "Enable Emscripten threads (-pthread + SharedArrayBuffer)" OFF)

# On Apple, default to Metal backend on. Use GAME_GGML_METAL to override.
if(APPLE)
    option(GAME_GGML_METAL "Enable ggml Metal backend" ON)
else()
    option(GAME_GGML_METAL "Enable ggml Metal backend" OFF)
endif()

# -----------------------------------------------------------------------------
# Emscripten: force off all the hardware backends + CLI + tests; turn on WASM.
# -----------------------------------------------------------------------------
if(EMSCRIPTEN)
    set(GAME_GGML_METAL        OFF CACHE BOOL "" FORCE)
    set(GAME_GGML_CUDA         OFF CACHE BOOL "" FORCE)
    set(GAME_GGML_VULKAN       OFF CACHE BOOL "" FORCE)
    set(GAME_GGML_BUILD_CLI    OFF CACHE BOOL "" FORCE)
    set(GAME_GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
    set(GAME_GGML_BUILD_WASM   ON  CACHE BOOL "" FORCE)
    # ggml itself: disable native / AVX / etc; WASM SIMD via -msimd128.
    set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_HBM           OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS              OFF CACHE BOOL "" FORCE)
    # WebAssembly SIMD128 (stable across Chrome/Safari/Firefox since 2022).
    add_compile_options(-msimd128)
    if(GAME_GGML_WASM_THREADS)
        # Global -pthread so ggml's own CPU backend compiles with threading.
        add_compile_options(-pthread)
        add_link_options(-pthread)
    endif()
endif()

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(MSVC)
    foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_cfg}" _CFG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
    add_compile_options(/utf-8)
endif()

if(MSVC)
    set(GAME_GGML_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(GAME_GGML_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
# Dependencies (ggml via FetchContent, plus optional Metal patch)
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Core library: game_ggml
# -----------------------------------------------------------------------------
# NOTE: Only backend.cpp is present in Task 1; further source files are added
# in subsequent tasks.  Header-only parts of the public API live in include/.
set(GAME_GGML_CORE_SOURCES
    src/backend.cpp
    src/gguf_io.cpp
    src/tensor_utils.cpp
    src/ops_basic.cpp
    src/ops_ffn.cpp
    src/ops_rope.cpp
    src/ops_attn.cpp
    src/ops_joint_attn.cpp
    src/model_encoder.cpp
    src/model_segmenter.cpp
    src/model_estimator.cpp
    src/model.cpp
    src/mel.cpp
    src/decode.cpp
    src/rng.cpp
    src/d3pm.cpp
)

add_library(game_ggml STATIC ${GAME_GGML_CORE_SOURCES})
add_library(game_ggml::game_ggml ALIAS game_ggml)

target_include_directories(game_ggml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ggml targets exported by the FetchContent'd ggml source tree.
# v0.11.0 provides `ggml` as the umbrella interface target.
target_link_libraries(game_ggml
    PUBLIC  ggml
    PRIVATE pocketfft
)

target_compile_options(game_ggml PRIVATE ${GAME_GGML_WARN_FLAGS})
target_compile_definitions(game_ggml PUBLIC
    GAME_GGML_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GAME_GGML_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GAME_GGML_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Propagate optional-backend feature macros so consumers can branch on them.
if(GAME_GGML_METAL)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_METAL=1)
endif()
if(GAME_GGML_CUDA)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_CUDA=1)
endif()
if(GAME_GGML_VULKAN)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_VULKAN=1)
endif()
# GGML_WEBGPU is a variable set by ggml's own CMake; propagate it so our
# backend.cpp can #include <ggml-webgpu.h> and enumerate "webgpu" in
# available_backends().
if(GGML_WEBGPU)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_WEBGPU=1)
    # ggml-webgpu's precompiled flash_attn_ext shader library only covers a
    # fixed set of (head_dim, block_m, block_n) tuples.  GAME uses head_dim=32
    # which isn't in that table — calling `ggml_flash_attn_ext` on this
    # backend trips `GGML_ASSERT(decisions.path != ... PATH_NONE)`.  Switch to
    # manual Q@K^T + softmax + @V in our attention ops for this build.
    target_compile_definitions(game_ggml PRIVATE GAME_GGML_NO_FLASH_ATTN=1)
endif()

# -----------------------------------------------------------------------------
# CLI executable
# -----------------------------------------------------------------------------
if(GAME_GGML_BUILD_CLI)
    add_executable(game_ggml_cli
        src/cli/main.cpp
        src/cli/wav_io.cpp
        src/cli/slicer.cpp
        src/cli/midi_writer.cpp
        src/cli/text_writer.cpp
    )
    target_link_libraries(game_ggml_cli PRIVATE
        game_ggml::game_ggml
        dr_wav
    )
    target_compile_options(game_ggml_cli PRIVATE ${GAME_GGML_WARN_FLAGS})
endif()

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(GAME_GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# WASM module (Emscripten only)
# -----------------------------------------------------------------------------
if(GAME_GGML_BUILD_WASM)
    add_executable(game_ggml_wasm
        src/wasm_bindings.cpp
        src/wasm_exports.cpp
        src/cli/slicer.cpp
    )
    target_link_libraries(game_ggml_wasm PRIVATE game_ggml::game_ggml)
    target_compile_options(game_ggml_wasm PRIVATE ${GAME_GGML_WARN_FLAGS})
    set_target_properties(game_ggml_wasm PROPERTIES
        OUTPUT_NAME "game_ggml"
        SUFFIX ".js"
    )

    set(_wasm_link_opts
        "-lembind"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sEXPORT_NAME=createGameGGML"
        "-sENVIRONMENT=web,worker,node"
        "-sINITIAL_MEMORY=64MB"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMAXIMUM_MEMORY=2GB"
        "-sSTACK_SIZE=2MB"
        "-sEXPORTED_FUNCTIONS=_malloc,_free,_gg_create_model,_gg_destroy_model,_gg_set_threads,_gg_sample_rate,_gg_slice,_gg_infer,_gg_free,_gg_version,_gg_ggml_version"
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF32,UTF8ToString"
        "-sFORCE_FILESYSTEM=1"
        "-sNO_EXIT_RUNTIME=1"
    )

    if(GAME_GGML_WASM_THREADS)
        # Threads require: -pthread at both compile + link, SharedArrayBuffer
        # (implied by -pthread), and a cross-origin-isolated page (COOP/COEP
        # headers sent by the host — see web-demo/serve.py or the COI service
        # worker for GitHub Pages).
        target_compile_options(game_ggml_wasm PRIVATE -pthread)
        # Also pass -pthread through ggml's own targets so its thread calls
        # compile.  ggml reads the `ggml-*` target link properties so we rely
        # on the global add_compile_options below being active as well.
        list(APPEND _wasm_link_opts
            "-pthread"
            "-sPTHREAD_POOL_SIZE=8"        # pre-allocate 8 WebWorkers; ggml picks how many to use via setNumThreads
            "-sSHARED_MEMORY=1"
        )
    endif()

    if(GGML_WEBGPU)
        # WebGPU calls (RequestAdapter, onSubmittedWorkDone, …) are exposed
        # through emdawnwebgpu's JSPI (WebAssembly.Suspending) shim.  For JS
        # entry points to suspend into WASM, they must be wrapped with
        # `WebAssembly.promising`.  Emscripten does NOT auto-wrap exports by
        # default — we have to list them explicitly here.  embind-bound
        # functions aren't in this list (runtime-registered), so the JS side
        # uses plain-C wrappers (gg_*) for everything that reaches WebGPU.
        #
        # Two notes:
        #  * JSPI_EXPORTS matches against raw WASM export keys, which have no
        #    underscore prefix (unlike EXPORTED_FUNCTIONS).  Drop the `_`.
        #  * Only list functions that actually suspend — promising wrapping
        #    turns them into async (returning Promise), so every caller must
        #    `await` them.  `gg_create_model` calls `ggml_backend_webgpu_init`
        #    (adapter/device request → suspends).  `gg_infer` calls
        #    `ggml_backend_graph_compute` (queue.onSubmittedWorkDone →
        #    suspends).  Everything else (slice/set_threads/sample_rate/free)
        #    stays synchronous.
        list(APPEND _wasm_link_opts
            "-sJSPI_EXPORTS=gg_create_model,gg_destroy_model,gg_infer"
        )
    endif()

    target_link_options(game_ggml_wasm PRIVATE ${_wasm_link_opts})
endif()

# -----------------------------------------------------------------------------
# Install (minimal; mainly so consumers can embed via add_subdirectory)
# -----------------------------------------------------------------------------
install(DIRECTORY include/game_ggml
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(TARGETS game_ggml
        EXPORT game_ggml-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "game_ggml ${PROJECT_VERSION} configuration:")
message(STATUS "  Metal .............. ${GAME_GGML_METAL}")
message(STATUS "  CUDA ............... ${GAME_GGML_CUDA}")
message(STATUS "  Vulkan ............. ${GAME_GGML_VULKAN}")
message(STATUS "  Build CLI .......... ${GAME_GGML_BUILD_CLI}")
message(STATUS "  Build Tests ........ ${GAME_GGML_BUILD_TESTS}")
message(STATUS "")
