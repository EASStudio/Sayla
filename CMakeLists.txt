cmake_minimum_required(VERSION 3.20)
project(Sayla C CXX) 

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SAYLA_ENABLE_VOICE "Build with whisper.cpp voice listening support" ON)

include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
    GIT_TAG        5.5
    GIT_PROGRESS   TRUE
)

FetchContent_MakeAvailable(raylib)

if (UNIX AND NOT APPLE) 
    find_package(OpenGL REQUIRED COMPONENTS EGL)
endif()

if (SAYLA_ENABLE_VOICE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        whispercpp
        GIT_REPOSITORY "https://github.com/ggml-org/whisper.cpp.git"
        GIT_TAG        master
        GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(whispercpp)

    set(MINIAUDIO_INCLUDE_DIR "${raylib_SOURCE_DIR}/src/external")
endif()

# Source files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
set(PROJECT_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/src/")

# Executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# No debug console
if(WIN32)
    if(MSVC)
        # MSVC:
        target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
    else()
        # MinGW:
        target_link_options(${PROJECT_NAME} PRIVATE -mwindows)
    endif()
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

if (SAYLA_ENABLE_VOICE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE VOICE_ENABLED)
    target_include_directories(${PROJECT_NAME} PRIVATE ${MINIAUDIO_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE whisper)

    if (UNIX AND NOT APPLE)
        target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl)
    endif()
endif()