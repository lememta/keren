add_executable(keren-sim keren-sim.cpp)

target_link_libraries(keren-sim PRIVATE keren)

install(TARGETS keren-sim DESTINATION bin)
