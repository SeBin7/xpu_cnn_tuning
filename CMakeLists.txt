# CMakeLists.txt (REPLACE WHOLE FILE)
cmake_minimum_required(VERSION 3.18)
project(xpu_fused_conv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Integer cache options (no ON/OFF booleans)
set(TILE_H         8   CACHE STRING "tile height")
set(TILE_W         16  CACHE STRING "tile width")
set(CTILE          32  CACHE STRING "Cin tile")
set(KTILE          8   CACHE STRING "Cout tile")
set(PAD_X          1   CACHE STRING "row padding (0/1)")
set(TS             8   CACHE STRING "tile size (oh/ow)")
set(CI_STEP        2   CACHE STRING "Cin loop step")
set(KO_STEP        1   CACHE STRING "Cout loop step")
set(USE_HALF_TILE  1   CACHE STRING "use half-tile in SLM (0/1)")
set(SUBGROUP_HINT  16  CACHE STRING "reqd_sub_group_size hint (0/8/16/32)")

# ---- Torch / Python
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# ---- Common flags
add_compile_options(-O3 -ffast-math -fsycl -fsycl-device-code-split=per_kernel -Wno-nan-infinity-disabled)

# ---- Custom fused kernel
add_library(custom_conv3x3_bn_relu_xpu SHARED
  cpp/custom_conv3x3_bn_relu_xpu.cpp
)
target_include_directories(custom_conv3x3_bn_relu_xpu PRIVATE
  ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
)
target_link_libraries(custom_conv3x3_bn_relu_xpu PRIVATE ${TORCH_LIBRARIES})
target_link_options(custom_conv3x3_bn_relu_xpu PRIVATE -fsycl)

# numeric macros to device
target_compile_definitions(custom_conv3x3_bn_relu_xpu PRIVATE
  TILE_H=${TILE_H} TILE_W=${TILE_W} CTILE=${CTILE} KTILE=${KTILE}
  PAD_X=${PAD_X}  TS=${TS} CI_STEP=${CI_STEP} KO_STEP=${KO_STEP}
  USE_HALF_TILE=${USE_HALF_TILE} SUBGROUP_HINT=${SUBGROUP_HINT}
  TORCH_EXTENSION_NAME=custom_conv3x3_bn_relu_xpu
)
target_compile_options(custom_conv3x3_bn_relu_xpu PRIVATE
  -O3 -ffast-math -fno-math-errno
  -fsycl-device-code-split=per_kernel
  -fsycl-unnamed-lambda
  -fsycl-id-queries-fit-in-int
)
target_link_options(custom_conv3x3_bn_relu_xpu PRIVATE -Wl,--no-as-needed)

# ---- (Optional) keep old sample buildable, but only touch it AFTER it's created
if (EXISTS ${CMAKE_SOURCE_DIR}/cpp/fused_conv3x3_bnfold_relu_usm_xpu.cpp)
  add_library(fused_conv3x3_bn_relu_xpu SHARED
    cpp/fused_conv3x3_bnfold_relu_usm_xpu.cpp
  )
  target_include_directories(fused_conv3x3_bn_relu_xpu PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(fused_conv3x3_bn_relu_xpu PRIVATE ${TORCH_LIBRARIES})
  target_compile_options(fused_conv3x3_bn_relu_xpu PRIVATE -fsycl)
  target_link_options(fused_conv3x3_bn_relu_xpu PRIVATE -fsycl)

  # (only now) numeric defs for legacy target
  target_compile_definitions(fused_conv3x3_bn_relu_xpu PRIVATE
    TILE_H=${TILE_H} TILE_W=${TILE_W} CTILE=${CTILE} KTILE=${KTILE}
    PAD_X=${PAD_X}  TS=${TS} CI_STEP=${CI_STEP} KO_STEP=${KO_STEP}
    USE_HALF_TILE=${USE_HALF_TILE} SUBGROUP_HINT=${SUBGROUP_HINT}
  )
endif()

# RPATH for Torch
set_target_properties(custom_conv3x3_bn_relu_xpu PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)
