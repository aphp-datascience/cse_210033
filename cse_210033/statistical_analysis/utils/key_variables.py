import time
from datetime import timedelta
from typing import List

import polars as pl
from loguru import logger


def compute_duration_after_event(
    event_df: pl.DataFrame,
    cohort_visit: pl.DataFrame,
    col_date: str,
    thresholds: List[float],
    sensibility_variables: List[str],
):
    start = time.time()
    index = sensibility_variables.copy() if sensibility_variables else []
    index.append("sub_cohort")
    # Merge event and cohort per patient
    tic = time.time()
    event_per_cohort_stay = cohort_visit.select(
        pl.col(
            [
                "visit_cohort_id",
                "person_id",
                "cohort_stay_end",
            ]
            + index
        )
    ).join_asof(
        event_df.select(pl.col(["visit_occurrence_id", "person_id", col_date]))
        .filter(pl.col(col_date).is_not_null())
        .sort(col_date),
        left_on="cohort_stay_end",
        right_on=col_date,
        by="person_id",
        strategy="forward",
        tolerance="{}d".format(max(thresholds)),
    )
    tac = time.time()
    logger.debug("Merge event and cohort: {} s", tac - tic)

    # Remove cohort stay in outcome
    tic = time.time()
    event_per_cohort_stay = event_per_cohort_stay.filter(
        ~(pl.col("visit_cohort_id") == pl.col("visit_occurrence_id"))
        & (pl.col(col_date).is_not_null())
    )
    tac = time.time()
    logger.debug("Remove cohort stay in outcome: {} s", tac - tic)

    # Compute time difference between event and cohort stay
    tic = time.time()
    event_per_cohort_stay = event_per_cohort_stay.with_columns(
        (pl.col(col_date) - pl.col("cohort_stay_end")).alias(
            "duration_after_cohort_stay"
        )
    )
    tac = time.time()
    logger.debug("Compute time difference: {} s", tac - tic)

    # Thresholds
    tic = time.time()
    outcome_per_threshold = []
    for threshold in thresholds:
        outcome = event_per_cohort_stay.clone()
        outcome = outcome.with_columns(
            pl.when(pl.col("duration_after_cohort_stay") <= timedelta(days=threshold))
            .then(True)
            .otherwise(False)
            .alias("has_event"),
            pl.lit("before_{}_days".format(threshold)).alias("threshold"),
        )
        outcome_per_threshold.append(outcome)

    result = pl.concat(outcome_per_threshold)
    tac = time.time()
    logger.debug("Concat thresholds: {} s", tac - tic)

    # Sum events
    tic = time.time()
    result = (
        result.groupby(index + ["threshold"])
        .agg([pl.col("has_event").sum().alias("n_events")])
        .with_columns(pl.col("n_events").cast(pl.Int32))
    )
    tac = time.time()
    logger.debug("Sum events: {} s", tac - tic)

    # Compute rate
    tic = time.time()
    result = result.join(
        cohort_visit.groupby(index).agg(
            pl.col("visit_cohort_id").count().alias("n_total")
        ),
        on=index,
    ).with_columns((pl.col("n_events") / pl.col("n_total")).alias("rate"))
    tac = time.time()
    logger.debug("Compute rate: {} s", tac - tic)
    end = time.time()
    logger.info("Compute duration: {} s", end - start)
    return result


def compute_event_during_cohort_stay(
    event_df: pl.DataFrame,
    cohort_visit: pl.DataFrame,
    sensibility_variables: List[str],
    **kwargs
):
    start = time.time()
    index = sensibility_variables.copy() if sensibility_variables else []
    index.append("sub_cohort")
    # Merge event and cohort per patient
    tic = time.time()
    event_df = (
        event_df.rename({"visit_occurrence_id": "visit_cohort_id"})
        .with_columns(pl.lit(True).alias("has_event"))
        .select(pl.col(["visit_cohort_id", "has_event"]))
        .unique()
    )
    result = cohort_visit.join(event_df, on="visit_cohort_id", how="left")
    tac = time.time()
    logger.debug("Merge event and cohort: {} s", tac - tic)

    # Compute rate
    tic = time.time()
    result = (
        result.groupby(index)
        .agg(
            [
                pl.col("has_event").sum().alias("n_events"),
                pl.col("visit_cohort_id").n_unique().alias("n_total"),
            ]
        )
        .with_columns(
            pl.col("n_events").fill_null(strategy="zero").cast(pl.Int32),
        )
        .with_columns((pl.col("n_events") / pl.col("n_total")).alias("rate"))
    )
    tac = time.time()
    logger.debug("Compute rate: {} s", tac - tic)
    end = time.time()
    logger.info("Compute during: {} s", end - start)
    return result


def compute_condition_incidence(
    event_df: pl.DataFrame,
    cohort_visit: pl.DataFrame,
    sensibility_variables: List[str],
    **kwargs
):
    start = time.time()
    index_time = "cohort_stay_start"

    # Merge event and cohort per patient
    tic = time.time()
    event_df = (
        event_df.rename({"visit_occurrence_id": "visit_cohort_id"})
        .select(pl.col(["visit_cohort_id"]))
        .unique()
    )
    result = cohort_visit.join(event_df, on="visit_cohort_id", how="inner")
    tac = time.time()
    logger.debug("Merge event and cohort: {} s", tac - tic)

    # Sum events
    tic = time.time()
    result = (
        result.sort(index_time)
        .groupby_dynamic(index_time, every="1w", by=sensibility_variables)
        .agg([pl.col("visit_cohort_id").n_unique().alias("n_events")])
        .with_columns(pl.col("n_events").cast(pl.Int32))
        .rename({index_time: "sub_cohort"})
    )
    tac = time.time()
    logger.debug("Sum events: {} s", tac - tic)

    # Compute max_event per-winter
    tic = time.time()
    result = result.with_columns(
        (pl.col("sub_cohort").dt.year() + (pl.col("sub_cohort").dt.month() > 8)).alias(
            "school_years"
        )
    )
    index = sensibility_variables.copy() if sensibility_variables else []
    result_max = (
        result.sort(["n_events", "sub_cohort"], descending=[True, False])
        .groupby(["school_years"] + index)
        .first()
        .rename({"n_events": "max_events"})
    )
    tac = time.time()
    logger.debug("Compute max_event per-winter: {} s", tac - tic)

    # Join to result
    tic = time.time()
    result = result.join(
        result_max, on=["sub_cohort", "school_years"] + index, how="left"
    )
    tac = time.time()
    logger.debug("Join to result: {} s", tac - tic)
    end = time.time()
    logger.info("Compute incidence: {} s", end - start)
    return result
