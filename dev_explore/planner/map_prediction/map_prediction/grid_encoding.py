"""Occupancy-grid <-> model-value encoding shared by every prediction backend.

Deliberately torch-free: the oracle backend needs these tables but must not pull
in the learned model's dependencies.
"""

import numpy as np


FREE_VALUE = np.uint8(255)
OCCUPIED_VALUE = np.uint8(0)
UNKNOWN_VALUE = np.uint8(100)

OCCUPANCY_TO_MODEL = np.full(256, UNKNOWN_VALUE, dtype=np.uint8)
OCCUPANCY_TO_MODEL[np.uint8(0)] = FREE_VALUE
OCCUPANCY_TO_MODEL[np.uint8(100)] = OCCUPIED_VALUE

MODEL_TO_OCCUPANCY = np.full(256, -1, dtype=np.int8)
MODEL_TO_OCCUPANCY[FREE_VALUE] = 0
MODEL_TO_OCCUPANCY[OCCUPIED_VALUE] = 100
