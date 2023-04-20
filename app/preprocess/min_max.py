def min_max(number: float) -> float:
    X_min = 0
    X_range = 5.0 - X_min
    X_std = (number - X_min) / X_range
    maxx = 1
    minn = 0
    X_scaled = X_std * (maxx - minn) + minn
    return X_scaled
