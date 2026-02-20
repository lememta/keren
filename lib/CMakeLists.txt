add_library(keren
  # Original interpreter/simulator code
  Interpreter.cpp
  StableHLOHandler.cpp
  LinalgHandler.cpp
  TensorValue.cpp
  Pipeline.cpp

  # New compiler code
  Compiler.cpp
  OpLowering.cpp
  Fusion.cpp

  # Op lowerings
  Lowerings/ElementwiseOps.cpp
  Lowerings/ContractionOps.cpp
  Lowerings/ReductionOps.cpp
)

target_include_directories(keren PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Collect all StableHLO static libraries
file(GLOB STABLEHLO_LIBS "${STABLEHLO_BUILD_DIR}/lib/*.a")

# Use linker group to resolve circular deps between StableHLO and MLIR libs
# Note: --start-group/--end-group only work with GNU ld, not Apple ld
if(NOT APPLE)
  set(START_GROUP "-Wl,--start-group")
  set(END_GROUP "-Wl,--end-group")
endif()

target_link_libraries(keren PUBLIC
  ${START_GROUP}
  # MLIR core
  MLIRParser
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRFuncAllExtensions
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransformUtils
  MLIRQuantDialect
  MLIRQuantUtils
  MLIRShapeDialect
  MLIRSCFDialect
  MLIRComplexDialect
  MLIRSparseTensorDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRRewrite
  MLIRSupport
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRFuncTransforms
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  LLVMSupport
  # All StableHLO libs
  ${STABLEHLO_LIBS}
  ${END_GROUP}
)
