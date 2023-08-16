from typing import Dict, List

import pandas as pd
from edsteva.probes.utils.filter_df import filter_valid_observations
from edsteva.utils.framework import get_framework, is_koalas, to
from edsteva.utils.typing import Data, DataFrame

from .filter_events import clean_date, filter_diag, filter_source


def prepare_care_site(
    data: Data,
):
    care_site = data.care_site[
        [
            "care_site_id",
            "care_site_type_source_value",
            "care_site_short_name",
            "place_of_service_source_value",
        ]
    ]
    care_site = care_site.rename(
        columns={
            "care_site_type_source_value": "care_site_level",
            "place_of_service_source_value": "service_type",
        }
    )

    return care_site


def prepare_visit_detail(
    data: Data,
    care_site: DataFrame,
):
    # Table extraction
    visit_detail = data.visit_detail[
        [
            "visit_detail_id",
            "visit_detail_start_datetime",
            "visit_occurrence_id",
            "visit_detail_type_source_value",
            "care_site_id",
            "row_status_source_value",
        ]
    ].rename(
        columns={
            "care_site_id": "detail_care_site_id",
            "visit_detail_type_source_value": "transfer_type",
        }
    )

    # Keep valid observations
    visit_detail = filter_valid_observations(
        table=visit_detail, table_name="visit_detail", valid_naming="Actif"
    )

    # Clean date
    visit_detail = clean_date(df=visit_detail, col_date="visit_detail_start_datetime")

    detail_care_site = care_site.rename(
        columns={
            "care_site_id": "detail_care_site_id",
            "care_site_level": "detail_care_site_level",
        }
    )
    visit_detail = visit_detail.merge(
        detail_care_site[["detail_care_site_id", "detail_care_site_level"]],
        on="detail_care_site_id",
    )
    return visit_detail


def prepare_condition_occurrence(
    data: Data,
    AREM_data: Data,
    visit_occurrence: DataFrame,
    condition_source_system: str,
    diag_regex: List[str],
):
    # Select ORBIS or AREM
    if condition_source_system == "AREM":
        condition_occurrence = _prepare_AREM_condition(
            AREM_data=AREM_data,
            visit_occurrence=visit_occurrence,
            condition_source_system=condition_source_system,
        )
    elif condition_source_system == "ORBIS":
        condition_occurrence = _prepare_ORBIS_condition(
            data=data, condition_source_system=condition_source_system
        )
    else:
        raise AttributeError(
            "condition_source_system must be 'AREM' or 'ORBIS' not {}".format(
                condition_source_system
            )
        )

    # Filter Diag
    condition_occurrence = filter_diag(
        condition_occurrence=condition_occurrence, diag_regex=diag_regex
    )
    return condition_occurrence


def prepare_ghm(
    AREM_data: Data,
    cmd_source_system: str,
    cmd: Dict[str, List[str]],
):
    arem_visit = AREM_data.orbis_visite_calc[["ids_eds", "ids_eds_crypt"]].rename(
        columns={"ids_eds_crypt": "encounter_num", "ids_eds": "visit_occurrence_id"}
    )
    ghm = (
        AREM_data.i2b2_observation_fact_ghm[
            ["encounter_num", "concept_cd", "sourcesystem_cd"]
        ]
        .rename(columns={"sourcesystem_cd": "cdm_source"})
        .merge(arem_visit, on="encounter_num", how="left")
        .drop(columns="encounter_num")
    )

    ghm = filter_source(table=ghm, source_system=cmd_source_system, table_name="ghm")
    ghm["GHM_code"] = ghm.concept_cd.str.split(":").str.get(1)
    cmd_map = pd.DataFrame.from_dict(cmd, orient="index").stack().reset_index(level=0)
    cmd_map.columns = ["CMD_code", "CMD"]
    cmd_map = to(get_framework(ghm), cmd_map)
    if is_koalas(cmd_map):
        cmd_map = cmd_map.spark.hint("broadcast")
    ghm["CMD_code"] = ghm.GHM_code.str.slice(stop=2)
    ghm = ghm.merge(cmd_map, on="CMD_code")
    return ghm.drop(columns="concept_cd").drop_duplicates(
        ["visit_occurrence_id", "GHM_code"]
    )


def prepare_visit_occurrence(
    data: Data,
    visit_source_system: str,
):
    # Table extraction

    visit_occurrence = data.visit_occurrence[
        [
            "visit_occurrence_id",
            "visit_occurrence_source_value",
            "person_id",
            "care_site_id",
            "visit_start_datetime",
            "visit_end_datetime",
            "visit_source_value",
            "stay_source_value",
            "cdm_source",
            "row_status_source_value",
        ]
    ].rename(
        columns={
            "visit_source_value": "stay_type",
            "stay_source_value": "stay_source",
        }
    )

    visit_occurrence = filter_valid_observations(
        table=visit_occurrence, table_name="visit", invalid_naming="supprimé"
    )
    visit_occurrence = filter_source(
        table=visit_occurrence, source_system=visit_source_system, table_name="visit"
    )

    # Clean date
    visit_occurrence = clean_date(df=visit_occurrence, col_date="visit_start_datetime")
    visit_occurrence = clean_date(df=visit_occurrence, col_date="visit_end_datetime")

    return visit_occurrence


def prepare_note(data: Data, note_source_system: str):
    note = data.note[
        [
            "note_id",
            "person_id",
            "visit_occurrence_id",
            "note_datetime",
            "note_class_source_value",
            "row_status_source_value",
            "note_text",
            "cdm_source",
        ]
    ].rename(columns={"note_class_source_value": "note_type"})
    # Keep valid observations
    note = filter_valid_observations(
        table=note, table_name="note", valid_naming="Actif"
    )
    note = note[~(note.note_text.isna())]

    # Keep source observations
    note = filter_source(
        table=note, source_system=note_source_system, table_name="note"
    )

    return note


def prepare_note_care_site(extra_data: Data, care_site: DataFrame):
    note_ref = extra_data.note_ref[
        [
            "note_id",
            "ufr_source_value",
            "us_source_value",
        ]
    ]
    care_site_ref = extra_data.care_site_ref[
        [
            "care_site_source_value",
            "care_site_id",
        ]
    ].rename(columns={"care_site_id": "detail_care_site_id"})

    note_ref = note_ref.melt(
        id_vars="note_id",
        value_name="care_site_source_value",
    )
    note_care_site = note_ref.merge(care_site_ref, on="care_site_source_value")
    detail_care_site = care_site[
        care_site.care_site_level == "Unité Fonctionnelle (UF)"
    ].rename(
        columns={
            "care_site_id": "detail_care_site_id",
            "care_site_level": "detail_care_site_level",
        }
    )
    note_care_site = note_care_site.merge(
        detail_care_site[["detail_care_site_id", "detail_care_site_level"]],
        on="detail_care_site_id",
    )[["note_id", "detail_care_site_id", "detail_care_site_level"]].drop_duplicates()
    return note_care_site


def prepare_person(data: Data, person_source_system: str):
    person = data.person[
        [
            "person_id",
            "birth_datetime",
            "death_datetime",
            "gender_source_value",
            "status_source_value",
            "cdm_source",
        ]
    ]

    # Keep valid person
    person = person[person["status_source_value"] == "Actif"].drop(
        columns=["status_source_value"]
    )

    # Keep source observations
    person = filter_source(
        table=person, source_system=person_source_system, table_name="person"
    )

    # Clean date
    person = clean_date(df=person, col_date="birth_datetime")
    person = clean_date(df=person, col_date="death_datetime")

    return person


def _prepare_ORBIS_condition(data: Data, condition_source_system: str):
    # Table extraction
    ORBIS_condition = data.condition_occurrence[
        [
            "visit_occurrence_id",
            "visit_detail_id",
            "cdm_source",
            "condition_status_source_value",
            "condition_source_value",
            "condition_start_datetime",
        ]
    ]

    # Keep source observations
    ORBIS_condition = filter_source(
        table=ORBIS_condition,
        source_system=condition_source_system,
        table_name="condition",
    )
    return ORBIS_condition


def _prepare_AREM_condition(
    AREM_data: Data,
    visit_occurrence: DataFrame,
    condition_source_system: str,
):
    # Table extraction
    AREM_visit = AREM_data.visit_occurrence[
        ["visit_occurrence_id", "visit_occurrence_source_value"]
    ]

    AREM_condition = AREM_data.condition_occurrence[
        [
            "visit_occurrence_id",
            "cdm_source",
            "condition_status_source_value",
            "condition_source_value",
            "condition_start_datetime",
        ]
    ]

    # Keep source observations
    AREM_condition = filter_source(
        table=AREM_condition,
        source_system=condition_source_system,
        table_name="condition",
    )

    # Add visit_occurrence_source_value
    AREM_condition = (
        AREM_visit[
            [
                "visit_occurrence_id",
                "visit_occurrence_source_value",
            ]
        ]
        .merge(
            AREM_condition,
            on="visit_occurrence_id",
            how="inner",
        )
        .drop(columns="visit_occurrence_id")
    )
    AREM_condition = AREM_condition.merge(
        visit_occurrence[["visit_occurrence_source_value", "visit_occurrence_id"]],
        on="visit_occurrence_source_value",
        how="inner",
    ).drop(columns="visit_occurrence_source_value")
    return AREM_condition
