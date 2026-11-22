# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "STREAM"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.stream_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%stream_libs", config.stream_libs_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

tool_dirs = [
    config.stream_tools_dir,
    config.circt_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "stream-opt",
]
if config.stream_has_plugin:
  tools.append("circt-opt")

llvm_config.add_tool_substitutions(tools, tool_dirs)

if not config.stream_has_plugin:
  config.available_features.add("no-stream-plugin")
