from datetime import datetime

import pandas as pd
import polars as pl
import statsmodels.api as sm
from edsteva.utils.typing import DataFrame


def add_mcd(cohort_visit: pl.DataFrame):
    cohort_visit_all = (
        cohort_visit.drop(["CMD", "CMD_code"])
        .clone()
        .with_columns(pl.lit("00 ALL").alias("MCD"))
    )
    cohort_visit_mcd = (
        cohort_visit.filter(
            (pl.col("CMD_code") <= "27")
            & (pl.col("sub_cohort") >= datetime(2016, 1, 1))
        )
        .with_columns((pl.col("CMD_code") + " " + pl.col("CMD")).alias("MCD"))
        .drop(["CMD", "CMD_code"])
    )

    return cohort_visit_mcd, cohort_visit_all


def add_care_site(cohort_visit: pl.DataFrame):
    cohort_visit = cohort_visit.with_columns(pl.col("care_site_id").cast(str))
    cohort_visit_all = (
        cohort_visit.drop(["care_site_id"])
        .clone()
        .with_columns(pl.lit("All").alias("care_site_id"))
    )
    cohort_visit = pl.concat([cohort_visit, cohort_visit_all], how="diagonal")

    return cohort_visit


def t_test(
    data: DataFrame,
    y_col: str,
    x_col: str,
    as_string: bool = False,
    **kwargs,
):
    index = list(
        {
            "MCD",
            "care_site_id",
            "threshold",
            "max_error",
            "start_observation_date",
            "Statistical analysis",
            "min_c_0",
            "outcome_name",
            "young_limit_age",
        }.intersection(data.columns)
    )

    # Fill na
    end_date = data.sub_cohort.max()
    filled_data = []
    for start_observation_date in data.start_observation_date.unique():
        date_index = pd.date_range(
            start=start_observation_date,
            end=end_date,
            freq="MS",
            closed=None,
        )
        date_index = pd.DataFrame({x_col: date_index})
        # Generate all available partitions
        all_partitions = (
            data[data.start_observation_date == start_observation_date][index]
            .drop_duplicates()
            .merge(date_index, how="cross")
        )
        filled_data.append(
            all_partitions.merge(
                data,
                on=[*index, x_col],
                how="left",
            ).fillna(0)
        )
    filled_data = pd.concat(filled_data)
    iter = filled_data.groupby(index, dropna=False)
    results = []
    for partition, group in iter:
        row = dict(zip(index, partition))
        p_value, alpha_0, alpha_1, mean_value = _compute_one_t_test(
            group, x_col, y_col, as_string
        )
        row["p_value"] = p_value
        row["alpha_0"] = alpha_0
        row["alpha_1"] = alpha_1
        row["mean_value"] = mean_value
        results.append(row)
    return pd.DataFrame(results)


def _compute_one_t_test(
    group: DataFrame,
    x_col: str,
    y_col: str,
    as_string: bool,
):
    X = list(range(group.shape[0]))
    try:
        X = sm.add_constant(X)
    except ValueError:
        p_value = "NOT SPECIFIED"
        alpha_0 = "NOT SPECIFIED"
        alpha_1 = "NOT SPECIFIED"
        return p_value, alpha_0, alpha_1
    y = group.sort_values(x_col)[y_col]
    model = sm.OLS(y.astype(float), X.astype(float))
    fitted_model = model.fit()
    if as_string:
        # p_value
        p_value = fitted_model.pvalues[1]
        if p_value < 0.001:
            p_value = "< 10e-3"
        else:
            p_value = f"{p_value:.3f}"

        # alpha_0
        alpha_0 = f"{fitted_model.params[0]:.2e}"
        ci_inf = f"{fitted_model.conf_int().iloc[0, 0]:.2e}"
        ci_sup = f"{fitted_model.conf_int().iloc[0, 1]:.2e}"
        alpha_0 = alpha_0 + " [" + ci_inf + ", " + ci_sup + "]"

        # alpha_1
        alpha_1 = f"{fitted_model.params[1]:.2e}"
        ci_inf = f"{fitted_model.conf_int().iloc[1, 0]:.2e}"
        ci_sup = f"{fitted_model.conf_int().iloc[1, 1]:.2e}"
        alpha_1 = alpha_1 + " [" + ci_inf + ", " + ci_sup + "]"
    else:
        # p_value
        p_value = fitted_model.pvalues[1]
        # alpha_0
        alpha_0 = fitted_model.params[0]
        # alpha_1
        alpha_1 = fitted_model.params[1]

    mean_value = group[y_col].mean()
    return p_value, alpha_0, alpha_1, mean_value
