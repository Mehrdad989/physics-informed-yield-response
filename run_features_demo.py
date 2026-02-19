from piyrm.features import generate_region_features

f = generate_region_features(seed=0, n_regions=5)
for i in range(5):
    print(
        int(f["region"][i]),
        round(float(f["aridity_index"][i]), 3),
        round(float(f["soil_whc"][i]), 1),
        round(float(f["heat_stress"][i]), 3),
    )
