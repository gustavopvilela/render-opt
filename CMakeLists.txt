cmake_minimum_required(VERSION 3.15...3.31)

project(render-opt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sa_core
        cpp/simulated_annealing.cpp
        cpp/bindings.cpp
)

target_include_directories(sa_core PRIVATE cpp/include)

if(MSVC)
        target_compile_options(sa_core PRIVATE /utf-8 /W4 /permissive-)
else()
    target_compile_options(sa_core PRIVATE -Wall -Wextra)
endif()

install(TARGETS sa_core DESTINATION .)