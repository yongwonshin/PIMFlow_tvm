if(USE_PIM)
  file(GLOB PIM_RELAY_CONTRIB_SRC src/relay/backend/contrib/pim/*.cc)
  list(APPEND COMPILER_SRCS ${PIM_RELAY_CONTRIB_SRC})

  message(STATUS "Build with PIM")
endif()
