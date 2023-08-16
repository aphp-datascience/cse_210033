import time
from datetime import datetime
from typing import List

import polars as pl
from loguru import logger

from .utils import key_functions
from .utils.complete_source import estimate_parameters, filter_unstable_cs_from_event_df


def statistical_analysis(
    cohort_visit_all: pl.DataFrame,
    cs_count: pl.DataFrame,
    event_df: pl.DataFrame,
    ehr_estimates: pl.DataFrame,
    event_name: str,
    key_function: str,
    care_site_level: str,
    start_observation_dates: List[str],
    end_date: str = None,
    sensibility_variables: List[str] = ["care_site_id"],
    thresholds: List[float] = None,
    col_date: str = None,
    stay_type: str = None,
    note_type: str = None,
    source_system: str = None,
    diag_type: str = None,
    specialties_set: str = None,
    condition_type: str = None,
    **kwargs
):
    # Estimates parameters
    naive_cs_count, max_errors_from_stats, min_c_0s_from_stats = estimate_parameters(
        ehr_estimates=ehr_estimates,
        care_site_level=care_site_level,
        stay_type=stay_type,
        note_type=note_type,
        source_system=source_system,
        diag_type=diag_type,
        specialties_set=specialties_set,
        condition_type=condition_type,
    )

    # Naive
    tic = time.time()
    naive_analysis_all = key_functions.get(key_function)(
        event_df=event_df,
        cohort_visit=cohort_visit_all,
        sensibility_variables=sensibility_variables,
        col_date=col_date,
        thresholds=thresholds,
    )
    naive_cs_count_all = cs_count.filter(
        pl.col("care_site_level") == care_site_level
    ).with_columns(
        pl.lit(naive_cs_count).alias(
            "naive_cs_count"
        ),  # Care site with at least one record
    )
    naive_analysis_with_params = []
    cs_count_with_params = []
    for start_observation_date in start_observation_dates:
        for max_error in max_errors_from_stats:
            for min_c_0 in min_c_0s_from_stats:
                naive_analysis_with_param = naive_analysis_all.clone()
                naive_analysis_with_param = naive_analysis_with_param.filter(
                    pl.col("sub_cohort")
                    >= datetime.strptime(start_observation_date, "%Y-%m-%d")
                ).with_columns(
                    pl.lit(start_observation_date).alias("start_observation_date"),
                    pl.lit(max_error).alias("max_error"),
                    pl.lit(min_c_0).alias("min_c_0"),
                )
                naive_analysis_with_params.append(naive_analysis_with_param)
                cs_count_with_params.append(
                    naive_cs_count_all.clone().with_columns(
                        pl.lit(start_observation_date).alias("start_observation_date"),
                        pl.lit(max_error).alias("max_error"),
                        pl.lit(min_c_0).alias("min_c_0"),
                    )
                )
    naive_analysis = pl.concat(naive_analysis_with_params).with_columns(
        pl.lit("Naive analysis").alias("Statistical analysis"),
    )
    naive_cs_count_summary = pl.concat(cs_count_with_params)
    tac = time.time()
    logger.debug("Naive analysis for {} is completed: {} s", event_name, tac - tic)

    # Complete-source-only
    tic = time.time()
    complete_source_only_analysis = []
    cso_cs_counts = []
    for start_observation_date in start_observation_dates:
        for max_error in max_errors_from_stats:
            for min_c_0 in min_c_0s_from_stats:
                cso_cs_count, stable_event_df = filter_unstable_cs_from_event_df(
                    event_df=event_df,
                    ehr_estimates=ehr_estimates,
                    care_site_level=care_site_level,
                    stay_type=stay_type,
                    note_type=note_type,
                    source_system=source_system,
                    diag_type=diag_type,
                    specialties_set=specialties_set,
                    condition_type=condition_type,
                    start_observation_date=start_observation_date,
                    end_date=end_date,
                    max_error=max_error,
                    min_c_0=min_c_0,
                )
                cso_analysis = key_functions.get(key_function)(
                    event_df=stable_event_df,
                    cohort_visit=cohort_visit_all,
                    sensibility_variables=sensibility_variables,
                    col_date=col_date,
                    thresholds=thresholds,
                )
                cso_analysis = cso_analysis.filter(
                    pl.col("sub_cohort")
                    >= datetime.strptime(start_observation_date, "%Y-%m-%d")
                ).with_columns(
                    pl.lit(start_observation_date).alias("start_observation_date"),
                    pl.lit(max_error).alias("max_error"),
                    pl.lit(min_c_0).alias("min_c_0"),
                    pl.lit("Complete-source-only analysis").alias(
                        "Statistical analysis"
                    ),
                )
                complete_source_only_analysis.append(cso_analysis)
                cso_cs_counts.append(
                    cs_count.filter(
                        pl.col("care_site_level") == care_site_level
                    ).with_columns(
                        pl.lit(cso_cs_count).alias("cso_cs_count"),
                        pl.lit(start_observation_date).alias("start_observation_date"),
                        pl.lit(max_error).alias("max_error"),
                        pl.lit(min_c_0).alias("min_c_0"),
                    )
                )
        logger.debug(
            "Complete-source-only analysis for {} at start observation date : {} is completed",
            event_name,
            start_observation_date,
        )

    complete_source_only_analysis = pl.concat(complete_source_only_analysis)
    cso_cs_count_summary = pl.concat(cso_cs_counts)
    tac = time.time()
    logger.debug(
        "Complete-source-only analysis for {} is completed: {} s",
        event_name,
        tac - tic,
    )

    # Concatenate result
    result = pl.concat(
        [
            naive_analysis,
            complete_source_only_analysis,
        ],
        how="diagonal",
    ).to_pandas()
    cs_count_outcome = cso_cs_count_summary.join(
        naive_cs_count_summary,
        on=["start_observation_date", "max_error", "min_c_0", "total_care_site"],
        how="inner",
    ).to_pandas()

    return cs_count_outcome, result
