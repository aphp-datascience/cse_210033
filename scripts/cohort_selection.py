import os
import sys

from edsteva import improve_performances
from edsteva.io import HiveData
from edsteva.utils.framework import is_koalas, to
from edstoolbox import SparkApp
from loguru import logger
from rich import print

from cse_210033 import BASE_DIR
from cse_210033.cohort_selection import cohort_selection
from cse_210033.utils import dump_data, timemeasure

improve_performances()
app = SparkApp("CSE210033 - Cohort Selection")


@app.submit
def run(spark, sql, config):
    timer = timemeasure()
    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    save_folder_path = BASE_DIR / "data" / "cohort_selection"
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
        print("the folder {} has been created".format(save_folder_path))
    load_data_conf = config["load_data"]
    print("Load data from database {}".format(load_data_conf["database_name"]))
    # Time measurement
    timer.lap(event_name="Setup config")

    data = HiveData(
        database_name=load_data_conf["database_name"],
        database_type=load_data_conf["database_type"],
        tables_to_load=load_data_conf["tables_to_load"],
        spark_session=spark,
    )
    prod_data = HiveData(
        database_name=load_data_conf["prod_database_name"],
        database_type=load_data_conf["prod_database_type"],
        tables_to_load=load_data_conf["prod_tables_to_load"],
        spark_session=spark,
    )
    AREM_data = HiveData(
        database_name=load_data_conf["AREM_database_name"],
        tables_to_load=load_data_conf["GHM_tables_to_load"],
        spark_session=spark,
    )
    # Time measurement
    timer.lap(event_name="Load data")

    # Count records in all EDS
    eds_count = dict(
        all_patient=data.person.person_id.nunique(),
        all_visit=data.visit_occurrence.visit_occurrence_id.nunique(),
        all_note=data.note.note_id.nunique(),
        patient_with_visit=data.visit_occurrence.person_id.nunique(),
    )
    dump_data(
        eds_count,
        BASE_DIR
        / "logs"
        / "cohort_selection"
        / sorted(os.listdir(BASE_DIR / "logs" / "cohort_selection"))[-1]
        / "eds_count.json",
    )
    timer.lap(event_name="Count records in all EDS")

    # Cohort selection
    cohort_selection_conf = config["cohort_selection"]
    outcomes = cohort_selection(
        data=data, prod_data=prod_data, AREM_data=AREM_data, **cohort_selection_conf
    )
    # Time measurement
    timer.lap(event_name="Cohort selection query")

    for outcome_name, outcome_df in outcomes.items():
        outcome_path = save_folder_path / "{}.pickle".format(outcome_name)
        outcome_name = outcome_name.replace("_", " ")
        print("Selecting {} stays...".format(outcome_name))
        if is_koalas(outcome_df):
            outcome_df = to("pandas", outcome_df)
        outcome_df.to_pickle(outcome_path)
        logger.info(
            "{} table has been saved in {}", outcome_name.capitalize(), outcome_path
        )
        logger.info(
            "{} table has shape {}", outcome_name.capitalize(), outcome_df.shape
        )
        del outcome_df
        # Time measurement
        timer.lap(event_name="Selecting {} stays".format(outcome_name))
    timer.stop(script_name="cohort_selection")

    print("Data has been preprocessed and saved ! :sunglasses:")


if __name__ == "__main__":
    app.run()
