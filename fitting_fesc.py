import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from astropy import units as u


def fit_model(df, lin_reg, X_all, mstar_cutoff=9):
    df["f_esc_fit"] = lin_reg.predict(X_all)
    return


def get_relative_error(fesc_df):
    relative_error = (
        np.abs(fesc_df.fesc_true - fesc_df.fesc_predict) / fesc_df.fesc_true
    )
    average_error = relative_error.mean()
    return average_error


def analyze_results(X_poly_test, y_test, lin_reg):
    y_predict = lin_reg.predict(X_poly_test)
    massive_halos = np.where(X_poly_test[:, 0] > 8)
    y_predict[massive_halos] = 0
    analysis_df = pd.DataFrame(
        {"fesc_true": y_test, "fesc_predict": y_predict}
    )
    low_fesc = analysis_df[analysis_df.fesc_true < 0.01]
    fraction_wrong = np.sum(low_fesc.fesc_predict > 0.01) / len(low_fesc)

    # print(np.sum(analysis_df.fesc_predict > 0.1) / len(analysis_df))

    heigher_fesc = analysis_df[analysis_df.fesc_true > 0.01]
    average_error = get_relative_error(heigher_fesc)
    print(analysis_df.describe())
    print(average_error)
    # fractions_wrong.append(fraction_wrong)
    # average_errors.append(average_error)
    return


def select_feature(n_tot_features, n_not_used):
    zero_indices = combinations(range(n_tot_features), n_not_used)
    arrays = [
        [0 if i in zero_indices else 1 for i in range(n_tot_features)]
        for zero_indices in zero_indices
    ]
    arrays = np.array(arrays)
    return arrays


def train_model(df, order=2):
    g_to_msun = (1 * u.g).to(u.M_sun)
    df["M_gas_sun_log"] = np.log10(df["M_gas"] * g_to_msun)
    df["M_star_sun_log"] = np.log10(df["M_star"] * g_to_msun)
    df.dropna(subset="f_esc", inplace=True)
    df.drop(df[df["M_star_sun_log"] < 5.55].index, inplace=True)

    train, test = train_test_split(df, test_size=0.1)

    feature_cols = ["M_star_sun_log", "M_gas_sun_log", "Metallicity", "z"]
    X = train.loc[:, feature_cols]
    y = train.f_esc
    X_test = test.loc[:, feature_cols]
    y_test = test.f_esc
    poly_features = PolynomialFeatures(degree=order, include_bias=False)

    # masks = select_feature(14, 8)
    # columns_used = []
    # fractions_wrong = []
    # average_errors = []
    # columns_to_use = np.where(masks[i, :] == 1)[0]
    columns_to_use = [0, 1, 4, 5, 8, 10]
    # columns_to_use = np.arange(14)
    X_poly = poly_features.fit_transform(X)
    X_poly_test = poly_features.fit_transform(X_test)

    X_poly = X_poly[:, columns_to_use]
    X_poly_test = X_poly_test[:, columns_to_use]
    print(poly_features.get_feature_names_out()[[0, 1, 4, 5, 8, 10]])
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    analyze_results(X_poly_test, y_test, lin_reg)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    # columns_used.append(columns_to_use)
    # print(len(fractions_wrong))
    # print(np.min(fractions_wrong))
    # print(average_errors[np.argmin(fractions_wrong)])
    # print(columns_used[np.argmin(fractions_wrong)])

    # print(np.min(average_errors))
    # print(fractions_wrong[np.argmin(average_errors)])
    # print(columns_used[np.argmin(average_errors)])
    return


def get_fesc_fit(df):
    g_to_msun = (1 * u.g).to(u.M_sun)
    df["M_gas_sun_log"] = np.log10(df["M_gas"] * g_to_msun)
    df["M_star_sun_log"] = np.log10(df["M_star"] * g_to_msun)

    fesc = (
        1.8589 * df["M_star_sun_log"]
        - 2.5041 * df["M_gas_sun_log"]
        + 0.1197 * df["M_star_sun_log"] ** 2
        - 0.4676 * df["M_star_sun_log"] * df["M_gas_sun_log"]
        + 0.3737 * df["M_gas_sun_log"] ** 2
        - 0.0019 * df["M_gas_sun_log"] * df["z"]
        + 3.562
    )
    df["f_esc_model"] = fesc
    df.loc[df["M_star_sun_log"] > 8.5, "f_esc_model"] = 0
    df.loc[df["f_esc_model"] < 0, "f_esc_model"] = 0
    df.loc[df["f_esc_model"] > 1, "f_esc_model"] = 1

    low_fesc = df[df.f_esc < 0.01]
    fraction_wrong = np.sum(low_fesc.f_esc_model > 0.01) / len(low_fesc)
    heigher_fesc = df[df.f_esc > 0.01]
    average_error = (
        np.abs(heigher_fesc.f_esc - heigher_fesc.f_esc_model)
        / heigher_fesc.f_esc
    ).mean()
    print(average_error)
    print(fraction_wrong)
    return
