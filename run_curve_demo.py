import numpy as np
from piyrm.curves import yield_rain_hump

# Define rainfall range (seasonal total in mm)
rain = np.linspace(0, 800, 9)

# Define parameters
ymax = 8.0       # t/ha
y_min = 1.0      # minimum yield
r_opt = 450.0    # optimal rainfall (mm)
width = 140.0    # tolerance (mm)

y = yield_rain_hump(
    rain_mm=rain,
    ymax=ymax,
    r_opt_mm=r_opt,
    width_mm=width,
    y_min=y_min,
)

for r, yy in zip(rain, y):
    print(f"Rain: {r:6.1f} mm | Yield: {yy:5.2f} t/ha")
