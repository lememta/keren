add_library(keren
  Interpreter.cpp
  StableHLOHandler.cpp
  LinalgHandler.cpp
  TensorValue.cpp
  Pipeline.cpp
)

target_include_directories(keren PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(keren PUBLIC
  MLIRParser
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRPass
  StablehloOps
  StablehloReferenceApi
  StablehloLinalgTransforms
)
