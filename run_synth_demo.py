from piyrm.synthetic import generate_synthetic_region_yield

d = generate_synthetic_region_yield(seed=42, n_regions=3, n_per_region=5)

# print first 5 rows
for i in range(5):
    print(
        d["region"][i],
        round(float(d["rain_mm"][i]), 2),
        round(float(d["yield_true"][i]), 3),
        round(float(d["yield_obs"][i]), 3),
    )
