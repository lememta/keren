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
target_link_libraries(keren PUBLIC
  -Wl,--start-group
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
  -Wl,--end-group
)
