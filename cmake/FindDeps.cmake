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

# Find StableHLO
find_package(StableHLO REQUIRED CONFIG)
message(STATUS "Using StableHLO in: ${StableHLO_DIR}")
include_directories(${STABLEHLO_INCLUDE_DIRS})
