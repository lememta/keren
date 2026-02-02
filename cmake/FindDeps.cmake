# Find MLIR/LLVM
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Find StableHLO — use build tree directly (no cmake config installed)
if(NOT STABLEHLO_BUILD_DIR)
  set(STABLEHLO_BUILD_DIR "" CACHE PATH "Path to StableHLO build directory")
endif()
if(NOT STABLEHLO_SRC_DIR)
  set(STABLEHLO_SRC_DIR "" CACHE PATH "Path to StableHLO source directory")
endif()

if(STABLEHLO_BUILD_DIR AND STABLEHLO_SRC_DIR)
  message(STATUS "Using StableHLO source: ${STABLEHLO_SRC_DIR}")
  message(STATUS "Using StableHLO build:  ${STABLEHLO_BUILD_DIR}")
  include_directories(${STABLEHLO_SRC_DIR})
  include_directories(${STABLEHLO_BUILD_DIR})
  link_directories(${STABLEHLO_BUILD_DIR}/lib)
else()
  # Fallback to find_package
  find_package(StableHLO REQUIRED CONFIG)
  message(STATUS "Using StableHLO in: ${StableHLO_DIR}")
  include_directories(${STABLEHLO_INCLUDE_DIRS})
endif()
