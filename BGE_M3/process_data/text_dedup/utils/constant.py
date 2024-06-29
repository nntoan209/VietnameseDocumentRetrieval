import numpy as np
from typing import Any
HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
    64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
    # 32, 16 bit config does not use a mersenne prime.
    # The original reason for using mersenne prime was speed.
    # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
    32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
    16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
}