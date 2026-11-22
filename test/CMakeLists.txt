llvm_canonicalize_cmake_booleans(
  STREAM_HAS_PLUGIN
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STREAM_TEST_DEPENDS
  FileCheck
  stream-opt
)
if(TARGET StreamPlugin)
  list(APPEND STREAM_TEST_DEPENDS StreamPlugin)
endif()

add_lit_testsuite(check-stream
  "Running the Stream dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${STREAM_TEST_DEPENDS}
)
set_target_properties(check-stream PROPERTIES FOLDER "Tests")

add_lit_testsuites(STREAM ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${STREAM_TEST_DEPENDS}
)
