cmake_minimum_required(VERSION 3.29)
project(KMeansCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_HOST_COMPILER  "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.36.32532/bin/Hostx64/x64/cl.exe")
set(CMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe")
set(CMAKE_MT "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/mt.exe")
set(CMAKE_LIBRARY_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64")
link_directories("C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64")

# Aggiungi l'eseguibile
add_executable(CUDA kernel.cu
        common/common.h
        common/common.cpp)

set_target_properties(CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Add the resources directory to the list of directories to be copied
add_custom_command(TARGET CUDA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:CUDA>/resources )