import lit.formats

config.name = "Keren"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__)

import os

# Find keren-sim binary relative to the test build directory
config.substitutions.append(
    ("%keren-sim", os.path.join(os.path.dirname(__file__), "..", "tools", "keren-sim"))
)
