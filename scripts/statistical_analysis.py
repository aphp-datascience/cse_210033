import os
import sys
import warnings

import pandas as pd
import polars as pl
import typer
from confection import Config
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from loguru import logger
from rich import print

from cse_210033 import BASE_DIR
from cse_210033.statistical_analysis import statistical_analysis
from cse_210033.statistical_analysis.utils.complete_source import (
    filter_unstable_cs_from_event_df,
)
from cse_210033.statistical_analysis.utils.supplementary_variables import (
    add_care_site,
    add_mcd,
)
from cse_210033.utils import dump_data, timemeasure

warnings.filterwarnings("ignore")


def main(config_name: str = "config.cfg"):
    timer = timemeasure()
    # Load config
    config_path = BASE_DIR / "conf" / config_name
    config = Config().from_disk(config_path, interpolate=True)
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    config = config["statistical_analysis"]
    thresholds = config["thresholds"]
    # Time measurement
    timer.lap(event_name="Setup config")

    # Load Models
    ehr_modeling_folder_path = BASE_DIR / "data" / "ehr_modeling"
    models_folder_path = ehr_modeling_folder_path / "models"
    visit_model_path = models_folder_path / "visit.pickle"
    icu_model_path = models_folder_path / "icu_rectangle.pickle"
    condition_model_per_visit_path = models_folder_path / "condition_per_visit.pickle"
    note_model_path = models_folder_path / "note.pickle"
    note_model_per_visit_path = models_folder_path / "note_per_visit.pickle"

    visit_model = StepFunction()
    visit_model.load(visit_model_path)

    icu_model = RectangleFunction()
    icu_model.load(icu_model_path)

    condition_model_per_visit = StepFunction()
    condition_model_per_visit.load(condition_model_per_visit_path)

    note_model = StepFunction()
    note_model.load(note_model_path)

    note_model_per_visit = StepFunction()
    note_model_per_visit.load(note_model_per_visit_path)

    ehr_estimates = dict(
        visit=visit_model.estimates,
        icu_rectangle=icu_model.estimates,
        condition=condition_model_per_visit.estimates,
        note=note_model.estimates,
        note_per_visit=note_model_per_visit.estimates,
    )
    # Time measurement
    timer.lap(event_name="Load models")

    # Load care site count per level
    cs_count = pl.from_pandas(
        pd.read_pickle(ehr_modeling_folder_path / "cs_count.pickle")
    )
    # Time measurement
    timer.lap(event_name="Load care site count")

    # Load cohort data
    cohort_path = BASE_DIR / "data" / "cohort_selection"
    cohort_visit = pl.from_pandas(pd.read_pickle(cohort_path / "cohort_visit.pickle"))
    outcomes = dict(
        hospit_visit=pl.from_pandas(
            pd.read_pickle(cohort_path / "hospit_visit.pickle")
        ),
        emergency_visit=pl.from_pandas(
            pd.read_pickle(cohort_path / "emergency_visit.pickle")
        ),
        consultation_note=pl.from_pandas(
            pd.read_pickle(cohort_path / "consultation_note.pickle")
        ),
        prescription_note=pl.from_pandas(
            pd.read_pickle(cohort_path / "prescription_note.pickle")
        ),
        bronchiolitis_condition=pl.from_pandas(
            pd.read_pickle(cohort_path / "bronchiolitis_condition.pickle")
        ),
        flu_condition=pl.from_pandas(
            pd.read_pickle(cohort_path / "flu_condition.pickle")
        ),
        gastroenteritis_condition=pl.from_pandas(
            pd.read_pickle(cohort_path / "gastroenteritis_condition.pickle")
        ),
        nasopharyngitis_condition=pl.from_pandas(
            pd.read_pickle(cohort_path / "nasopharyngitis_condition.pickle")
        ),
        icu_visit=pl.from_pandas(pd.read_pickle(cohort_path / "icu_visit.pickle")),
        icu_visit_rectangle=pl.from_pandas(
            pd.read_pickle(cohort_path / "icu_visit.pickle")
        ),
    )
    # Time measurement
    timer.lap(event_name="Load cohort data")

    # Filter cohort visit
    cohort_visit = cohort_visit.select(
        [
            pl.col("visit_cohort_id"),
            pl.col("person_id"),
            pl.col("care_site_id"),
            pl.col("cohort_stay_start"),
            pl.col("cohort_stay_end"),
            pl.col("stay_type"),
            pl.col("CMD_code"),
            pl.col("CMD"),
            pl.col("sub_cohort"),
        ]
    ).filter(pl.col("cohort_stay_end").is_not_null())
    cohort_cs_count, cohort_visit = filter_unstable_cs_from_event_df(
        event_df=cohort_visit,
        ehr_estimates=pl.from_pandas(ehr_estimates["visit"]),
        start_observation_date=config["cohort_start_date"],
        care_site_level="Hôpital",
        stay_type="hospitalisés",
        visit_col="visit_cohort_id",
    )
    cohort_visit = add_care_site(cohort_visit=cohort_visit)
    _, cohort_visit_all = add_mcd(cohort_visit=cohort_visit)
    cohort_visit_all = cohort_visit_all.sort("cohort_stay_end")
    # Time measurement
    timer.lap(event_name="Filter cohort stays")

    # Outcomes
    cs_count_outcomes = []
    for outcome_name, outcome_df in outcomes.items():
        outcome_config = config[outcome_name]
        cs_count_outcome, result_outcome = statistical_analysis(
            cohort_visit_all=cohort_visit_all,
            cs_count=cs_count,
            event_df=outcome_df,
            thresholds=thresholds,
            ehr_estimates=pl.from_pandas(
                ehr_estimates[outcome_config["ehr_functionality"]]
            ),
            **outcome_config,
        )
        cs_count_outcome["outcome_name"] = outcome_name
        cs_count_outcomes.append(cs_count_outcome)
        result_outcome.to_pickle(
            BASE_DIR / "data" / "statistical_analysis" / "{}.pkl".format(outcome_name)
        )
        print("{} has been saved".format(outcome_name))
        # Time measurement
        timer.lap(
            event_name="Statistical analysis for {}".format(
                outcome_config["event_name"]
            )
        )
    cs_count_summary = pd.concat(cs_count_outcomes)
    cs_count_summary["cohort_hospital_count"] = cohort_cs_count
    cs_count_summary.to_pickle(
        BASE_DIR / "data" / "statistical_analysis" / "cs_count_summary.pkl"
    )
    timer.stop(script_name="statistical_analysis", create_folder=True)
    dump_data(
        config,
        BASE_DIR
        / "logs"
        / "statistical_analysis"
        / sorted(os.listdir(BASE_DIR / "logs" / "statistical_analysis"))[-1]
        / "config.json",
    )
    print("Data is post_processed and ready for stats_analysis ! :sunglasses:")


if __name__ == "__main__":
    typer.run(main)
