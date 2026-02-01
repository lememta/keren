configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  COPYONLY
)

# Copy test files to build directory
file(GLOB_RECURSE TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/stablehlo/*.mlir")
foreach(test_file ${TEST_FILES})
  file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${test_file})
  configure_file(${test_file} ${CMAKE_CURRENT_BINARY_DIR}/${rel_path} COPYONLY)
endforeach()

# Add a custom target to run lit tests
add_custom_target(check-keren
  COMMAND lit -v ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS keren-sim
  COMMENT "Running Keren lit tests"
)
