import time
from datetime import datetime

import polars as pl
from loguru import logger


def estimate_parameters(
    ehr_estimates: pl.DataFrame,
    care_site_level: str,
    note_type: str = None,
    stay_type: str = None,
    specialties_set: str = None,
    diag_type: str = None,
    source_system: str = None,
    condition_type: str = None,
):
    start = time.time()
    # Filter estimates
    ehr_estimates = filter_estimates(
        ehr_estimates=ehr_estimates,
        care_site_level=care_site_level,
        note_type=note_type,
        stay_type=stay_type,
        specialties_set=specialties_set,
        diag_type=diag_type,
        source_system=source_system,
        condition_type=condition_type,
        renaming=False,
    )

    active_cs_count = ehr_estimates.select(
        "care_site_id"
    ).n_unique()  # Care site with at least one record on the study period

    max_errors = [1.0]
    max_errors.append(ehr_estimates.select(pl.col("error").quantile(0.75)).item())
    max_errors.append(ehr_estimates.select(pl.col("error").quantile(0.5)).item())
    max_errors.append(ehr_estimates.select(pl.col("error").quantile(0.25)).item())
    min_c_0s = [0.0]
    min_c_0s.append(ehr_estimates.select(pl.col("c_0").quantile(0.25)).item())
    min_c_0s.append(ehr_estimates.select(pl.col("c_0").quantile(0.5)).item())
    min_c_0s.append(ehr_estimates.select(pl.col("c_0").quantile(0.75)).item())
    end = time.time()
    logger.debug(
        "Estimate max errors ({}) and min c0 ({}): {} s",
        max_errors,
        min_c_0s,
        end - start,
    )
    return active_cs_count, max_errors, min_c_0s


def filter_unstable_cs_from_event_df(
    event_df: pl.DataFrame,
    ehr_estimates: pl.DataFrame,
    start_observation_date: str,
    care_site_level: str,
    max_error: float = 1.0,
    min_c_0: float = 0.0,
    note_type: str = None,
    stay_type: str = None,
    specialties_set: str = None,
    diag_type: str = None,
    source_system: str = None,
    condition_type: str = None,
    end_date: str = None,
    visit_col: str = "visit_occurrence_id",
):
    start = time.time()
    # Filter estimates
    ehr_estimates = filter_estimates(
        ehr_estimates=ehr_estimates,
        care_site_level=care_site_level,
        note_type=note_type,
        stay_type=stay_type,
        specialties_set=specialties_set,
        diag_type=diag_type,
        source_system=source_system,
        condition_type=condition_type,
    )

    # Filter unstable cs
    stable_cs = filter_unstable_cs_from_estimates(
        ehr_estimates=ehr_estimates,
        start_observation_date=start_observation_date,
        max_error=max_error,
        min_c_0=min_c_0,
        end_date=end_date,
    )

    # Groupby per visit
    tic = time.time()
    if care_site_level in ["Unité Fonctionnelle (UF)", "Unité d’hébergement (UH)"]:
        col_care_site = "detail_care_site_id"
        if "detail_care_site_level" in event_df.columns:
            event_df = event_df.filter(
                pl.col("detail_care_site_level") == care_site_level
            )
    elif care_site_level == "Hôpital":
        col_care_site = "care_site_id"
    else:
        return ValueError(
            "Argument care_site_level must be one of the following : ['Unité Fonctionnelle (UF)', 'Unité d’hébergement (UH)', 'Hôpital']"
        )

    # Count stable cs
    stable_cs_count = (
        stable_cs.filter(pl.col("stable_cs")).select(col_care_site).n_unique()
    )

    stable_cs = (
        event_df.with_columns(pl.col(col_care_site).cast(pl.Int64))
        .select(pl.col([visit_col, col_care_site]))
        .join(
            stable_cs.with_columns(pl.col(col_care_site).cast(pl.Int64)).select(
                pl.col(["stable_cs", col_care_site])
            ),
            on=col_care_site,
            how="left",
        )
        .with_columns(pl.col("stable_cs").fill_null(pl.lit(False)))
        .groupby(visit_col)
        .agg([pl.col("stable_cs").min()])
        .filter(pl.col("stable_cs"))
        .drop("stable_cs")
    )
    tac = time.time()
    logger.debug("Groupby per visit: {} s", tac - tic)

    # Filter on events
    tic = time.time()
    event_df = event_df.join(stable_cs, on=visit_col, how="inner")
    tac = time.time()
    end = time.time()
    logger.debug("Filter on events: {} s", tac - tic)
    logger.info("Filter {} on {} level: {} s", visit_col, care_site_level, end - start)
    return stable_cs_count, event_df


def filter_unstable_cs_from_estimates(
    ehr_estimates: pl.DataFrame,
    start_observation_date: str,
    max_error: float = 1.0,
    min_c_0: float = 0.0,
    end_date: str = None,
    **kwargs
):
    # Filter unstable cs
    tic = time.time()
    stable_cs = ehr_estimates.with_columns(
        pl.when(
            (pl.col("t_0") <= datetime.strptime(start_observation_date, "%Y-%m-%d"))
            & (pl.col("error") <= max_error)
            & (pl.col("c_0") >= min_c_0)
        )
        .then(True)
        .otherwise(False)
        .alias("stable_cs")
    )
    if "t_1" in stable_cs.columns and end_date:
        stable_cs = stable_cs.with_columns(
            pl.when(
                (pl.col("stable_cs"))
                & (pl.col("t_1") >= datetime.strptime(end_date, "%Y-%m-%d"))
            )
            .then(True)
            .otherwise(False)
            .alias("stable_cs")
        )
    tac = time.time()
    logger.debug("Filter unstable cs: {} s", tac - tic)
    return stable_cs


def filter_estimates(
    ehr_estimates: pl.DataFrame,
    care_site_level: str,
    note_type: str = None,
    stay_type: str = None,
    specialties_set: str = None,
    diag_type: str = None,
    source_system: str = None,
    condition_type: str = None,
    renaming: bool = True,
    **kwargs
):
    # Filter Estimates
    tic = time.time()
    ehr_estimates = ehr_estimates.filter(pl.col("care_site_level") == care_site_level)
    if note_type:
        ehr_estimates = ehr_estimates.filter(pl.col("note_type") == note_type)
    if stay_type:
        ehr_estimates = ehr_estimates.filter(pl.col("stay_type") == stay_type)
    if specialties_set:
        ehr_estimates = ehr_estimates.filter(
            pl.col("specialties_set") == specialties_set
        )
    if diag_type:
        ehr_estimates = ehr_estimates.filter(pl.col("diag_type") == diag_type)
    if source_system:
        ehr_estimates = ehr_estimates.filter(pl.col("source_system") == source_system)
    if condition_type:
        ehr_estimates = ehr_estimates.filter(pl.col("condition_type") == condition_type)
    if renaming and care_site_level in [
        "Unité Fonctionnelle (UF)",
        "Unité d’hébergement (UH)",
    ]:
        ehr_estimates = ehr_estimates.rename({"care_site_id": "detail_care_site_id"})
    tac = time.time()
    logger.debug("Filter Estimates: {} s", tac - tic)
    return ehr_estimates
