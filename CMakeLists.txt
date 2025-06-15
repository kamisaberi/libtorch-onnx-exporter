#cmake_minimum_required(VERSION 3.31)
#project(onnx_export_import)
#
#set(CMAKE_CXX_STANDARD 20)
#
#list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
#find_package(Torch REQUIRED)
#find_package(onnxruntime REQUIRED)
#add_executable(onnx_export_import main.cpp)
#target_link_libraries(onnx_export_import "${TORCH_LIBRARIES}" ${ONNXRUNTIME_LIBRARY})


cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(OnnxToolchain)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
# --- Use pkg-config to find system libraries ---
# This is the standard, correct way on Linux.
find_package(PkgConfig REQUIRED)

# Find Protobuf. This will set PC_PROTOBUF_LIBRARIES and PC_PROTOBUF_INCLUDE_DIRS.
# It will automatically include Abseil dependencies.
pkg_check_modules(PC_PROTOBUF REQUIRED protobuf)

# --- Generate C++ from onnx.proto3 ---
find_package(Protobuf REQUIRED) # We still need this to find the protoc compiler
set(PROTO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3")
set(GENERATED_H "${CMAKE_CURRENT_BINARY_DIR}/onnx.proto3.pb.h")
set(GENERATED_CC "${CMAKE_CURRENT_BINARY_DIR}/onnx.proto3.pb.cc")

add_custom_command(
        OUTPUT "${GENERATED_H}" "${GENERATED_CC}"
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROTO_SRC}"
        DEPENDS "${PROTO_SRC}"
)

# --- Library for our generated Protobuf code ---
add_library(onnx_proto_lib STATIC
        "${GENERATED_CC}"
        "${GENERATED_H}"
)
target_include_directories(onnx_proto_lib PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}"
        ${PC_PROTOBUF_INCLUDE_DIRS} # Use include dirs from pkg-config
)
# We must link this intermediate library to the protobuf libs
target_link_libraries(onnx_proto_lib PUBLIC ${PC_PROTOBUF_LIBRARIES})

# ======================================================================
# TARGET 1: The Weight/Arch Serializer (Uses LibTorch)
# ======================================================================
find_package(Torch REQUIRED)
add_executable(serializer_for_onnx serializer_for_onnx.cpp)
target_link_libraries(serializer_for_onnx PRIVATE
        "${TORCH_LIBRARIES}"
)
target_include_directories(serializer_for_onnx PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# ======================================================================
# TARGET 2: The Custom ONNX Exporter (NO LibTorch)
# ======================================================================
add_executable(custom_onnx_exporter custom_onnx_exporter.cpp)

# Link to our generated library. Because the link to PC_PROTOBUF_LIBRARIES
# in onnx_proto_lib was PUBLIC, CMake passes that linkage on to this target.
target_link_libraries(custom_onnx_exporter PRIVATE onnx_proto_lib)

target_include_directories(custom_onnx_exporter PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)



# ======================================================================
# TARGET 3: The ONNX Importer/Runner
# ======================================================================
# --- THIS IS THE FIX ---
# We list ALL source files needed for the executable.
add_executable(onnx_importer
        onnx_importer.cpp
        # Add the CXX API implementation source file directly.
#        ${onnxruntime_SOURCE_DIR}/core/session/onnxruntime_cxx_api.cc
)

# Link to the main ONNX Runtime library.
target_link_libraries(onnx_importer PRIVATE ${ONNXRUNTIME_LIBRARY} onnxruntime)