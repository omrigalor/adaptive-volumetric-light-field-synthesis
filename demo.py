from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from volumetric_light import LightFieldOptimizer


def disk(shape, center, radius):
    yy, xx = np.indices(shape)
    return (((xx - center[1]) ** 2 + (yy - center[0]) ** 2) <= radius**2).astype(float)


targets = [disk((64, 64), (23, 23), 8), disk((64, 64), (42, 42), 9)]
result = LightFieldOptimizer(phase_levels=128).optimize(targets, [0.08, 0.13], iterations=35)
print(f"initial weighted error: {result.loss_history[0]:.5f}")
print(f"final weighted error:   {result.loss_history[-1]:.5f}")
for index, metric in enumerate(result.metrics, 1):
    print(f"plane {index}: {metric}")
