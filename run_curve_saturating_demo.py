import numpy as np
from piyrm.curves import yield_rain_saturating

rain = np.array([0, 50, 100, 200, 400, 600, 800], dtype=float)

y = yield_rain_saturating(rain_mm=rain, ymax=10.0, k=0.004, y_min=1.0)

for r, yy in zip(rain, y):
    print(f"Rain: {r:6.1f} mm | Yield: {yy:5.2f} t/ha")