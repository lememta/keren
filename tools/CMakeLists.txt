# keren-sim: StableHLO interpreter/simulator
add_executable(keren-sim keren-sim.cpp)
target_link_libraries(keren-sim PRIVATE keren)
install(TARGETS keren-sim DESTINATION bin)

# keren-compile: StableHLO to Linalg compiler
add_executable(keren-compile keren-compile.cpp)
target_link_libraries(keren-compile PRIVATE keren)
install(TARGETS keren-compile DESTINATION bin)
