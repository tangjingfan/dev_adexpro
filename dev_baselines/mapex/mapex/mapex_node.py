"""Entry point wrapper for the MapEx ROS2 node.

Adds MapEx/scripts/ (and its lama/ sibling) to sys.path so that the node
module's local imports resolve correctly regardless of how colcon installs it.
"""

import os
import sys

_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "MapEx", "scripts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from explore_visvarprob_node import main  # noqa: E402

__all__ = ["main"]
