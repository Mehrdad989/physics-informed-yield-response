import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


FEATURES = [
    "log_precip_mean",
    "log_precip_med",
    "precip_std",
    "precip_cv",
    "yield_mean",
    "yield_std",
    "yield_cv",
]


def fit_and_report(df, crop_name):
    X = df[FEATURES].values
    y_ymax = df["log_ymax"].values
    y_k = df["log_k"].values

    m1 = LinearRegression().fit(X, y_ymax)
    m2 = LinearRegression().fit(X, y_k)

    r2_ymax = r2_score(y_ymax, m1.predict(X))
    r2_k = r2_score(y_k, m2.predict(X))

    print(f"\n{crop_name.upper()} v2 RESULTS")
    print("R2 log(ymax):", round(r2_ymax, 3))
    print("R2 log(k):   ", round(r2_k, 3))


def main():
    maize = pd.read_csv("data/processed/ml_state_maize_v2.csv")
    wheat = pd.read_csv("data/processed/ml_state_wheat_v2.csv")

    # drop any rows with missing features (should be none, but safe)
    maize = maize.dropna(subset=FEATURES + ["log_ymax", "log_k"])
    wheat = wheat.dropna(subset=FEATURES + ["log_ymax", "log_k"])

    fit_and_report(maize, "maize")
    fit_and_report(wheat, "wheat")


if __name__ == "__main__":
    main()