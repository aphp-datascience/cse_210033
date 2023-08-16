from typing import List

import pandas as pd
from edsteva.utils.typing import DataFrame
from loguru import logger


def filter_event(
    df: DataFrame,
    col_to_filter: str,
    event_regex: str,
):
    df = df[
        df[col_to_filter].str.contains(event_regex, case=False, na=False, regex=True)
    ]

    logger.debug(
        "Filter event: the following regex {} has been applied on column {}.",
        event_regex,
        col_to_filter,
    )
    return df


def filter_source(table: DataFrame, source_system: str, table_name: str = None):
    table = table[table.cdm_source == source_system].drop(columns="cdm_source")
    logger.debug(
        "Filter source: {} source system has been selected for table {}.",
        source_system,
        table_name,
    )
    return table


def clean_date(df: DataFrame, col_date: str):
    df = df[~((df[col_date] < pd.Timestamp.min) | (df[col_date] > pd.Timestamp.max))]
    df[col_date] = df[col_date].astype("datetime64")
    return df


def filter_date(
    cohort_visit: DataFrame,
    start_date: str,
    end_date: str,
):
    # Filter on study period
    cohort_visit = cohort_visit[
        (cohort_visit["cohort_stay_start"] >= start_date)
        & (cohort_visit["cohort_stay_start"] < end_date)
    ]

    # Divide subcohort
    cohort_visit["sub_cohort"] = (
        cohort_visit.cohort_stay_start.astype("datetime64")
        .dt.strftime("%Y-%m")
        .astype("datetime64")
    )
    logger.debug(
        "Filter date: cohort filtered between {} and {}.",
        start_date,
        end_date,
    )

    return cohort_visit


def filter_first_event(df: DataFrame, col_date: str):
    # Keep first event of each visit
    df = (
        df.sort_values(col_date, ascending=True)
        .groupby("visit_occurrence_id", as_index=False)
        .first()
    )
    return df


def filter_note_type(note: DataFrame, note_type_regex: str):
    note = note[
        note.note_type.str.contains(note_type_regex, case=False, na=False, regex=True)
    ]
    logger.debug(
        "Filter note type: the following note type regex {} has been applied.",
        note_type_regex,
    )
    return note


def filter_diag(condition_occurrence: DataFrame, diag_regex: List[str]):
    # Filter diagnostics
    condition_occurrence = condition_occurrence.rename(
        columns={"condition_status_source_value": "diag_type"}
    )
    condition_occurrence = condition_occurrence[
        condition_occurrence.diag_type.str.contains(diag_regex)
    ]
    logger.debug(
        "Filter diag: the following stay types {} have been selected.", diag_regex
    )
    return condition_occurrence
