import os
import sys

from edsteva import improve_performances
from edsteva.io import HiveData
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from edsteva.probes import ConditionProbe, NoteProbe, VisitProbe
from edsteva.utils.framework import to
from edstoolbox import SparkApp
from loguru import logger
from rich import print

from cse_210033 import BASE_DIR
from cse_210033.utils import timemeasure

improve_performances()
app = SparkApp("CSE210033 - EHR Modeling")


@app.submit
def run(spark, _, config):
    timer = timemeasure()

    if config["debug"]["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    save_folder_path = BASE_DIR / "data" / "ehr_modeling"
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
        print("the folder {} has been created".format(save_folder_path))
    probes_folder_path = save_folder_path / "probes"
    models_folder_path = save_folder_path / "models"
    if not os.path.isdir(probes_folder_path):
        os.mkdir(probes_folder_path)
        print("the folder {} has been created".format(probes_folder_path))
    if not os.path.isdir(models_folder_path):
        os.mkdir(models_folder_path)
        print("the folder {} has been created".format(models_folder_path))

    visit_probe_path = probes_folder_path / "visit.pickle"
    icu_probe_path = probes_folder_path / "icu_rectangle.pickle"
    condition_probe_path = probes_folder_path / "condition.pickle"
    condition_probe_per_visit_path = probes_folder_path / "condition_per_visit.pickle"
    note_probe_path = probes_folder_path / "note.pickle"
    note_probe_per_visit_path = probes_folder_path / "note_per_visit.pickle"
    visit_model_path = models_folder_path / "visit.pickle"
    icu_model_path = models_folder_path / "icu_rectangle.pickle"
    condition_model_path = models_folder_path / "condition.pickle"
    condition_per_visit_model_path = models_folder_path / "condition_per_visit.pickle"
    note_model_path = models_folder_path / "note.pickle"
    note_per_visit_model_path = models_folder_path / "note_per_visit.pickle"
    # Time measurement
    timer.lap(event_name="Setup config")

    # Load data
    load_data_conf = config["load_data"]
    logger.info(
        "Loading main data from database {}...", load_data_conf["database_name"]
    )
    data = HiveData(
        database_name=load_data_conf["database_name"],
        database_type=load_data_conf["database_type"],
        tables_to_load=load_data_conf["tables_to_load"],
        spark_session=spark,
    )
    logger.info(
        "Loading prod data from database {} to link notes with UF...",
        load_data_conf["prod_database_name"],
    )
    prod_data = HiveData(
        database_name=load_data_conf["prod_database_name"],
        database_type=load_data_conf["prod_database_type"],
        tables_to_load=load_data_conf["prod_tables_to_load"],
        spark_session=spark,
    )
    logger.info(
        "Loading AREM data from database {} to get AREM conditions...",
        load_data_conf["AREM_database_name"],
    )
    AREM_data = HiveData(
        database_name=load_data_conf["AREM_database_name"],
        database_type=load_data_conf["AREM_database_type"],
        tables_to_load=load_data_conf["AREM_tables_to_load"],
        spark_session=spark,
    )
    # Time measurement
    timer.lap(event_name="Load data")

    # Compute probes
    ehr_conf = config["ehr_modeling"]

    # Count number of care sites
    care_site = data.care_site[["care_site_id", "care_site_type_source_value"]].rename(
        columns={"care_site_type_source_value": "care_site_level"}
    )
    care_site = care_site[
        ~(
            care_site.care_site_id.isin(ehr_conf["hospital_to_remove"])
        )  # Remove technical hospitals without patient
    ]
    cs_count = to(
        "pandas",
        care_site.groupby("care_site_level", as_index=False, dropna=False)
        .agg({"care_site_id": "nunique"})
        .rename(columns={"care_site_id": "total_care_site"}),
    )
    cs_count.to_pickle(save_folder_path / "cs_count.pickle")
    # Time measurement
    timer.lap(event_name="Count number of care sites per level")

    # Compute visit probe
    logger.info("Computing visit probe...")
    visit_probe = VisitProbe()
    visit_probe.compute(data=data, **ehr_conf["visit"])
    logger.info("Visit probe has shape {}", visit_probe.predictor.shape)
    visit_probe.save(visit_probe_path)
    logger.info("Visit probe saved in {}", visit_probe_path)
    # Time measurement
    timer.lap(event_name="Compute visit probe")

    # Compute ICU probe
    logger.info("Computing ICU probe...")
    icu_probe = VisitProbe()
    icu_probe.compute(data=data, **ehr_conf["icu"])
    logger.info("ICU probe has shape {}", icu_probe.predictor.shape)
    icu_probe.save(icu_probe_path)
    logger.info("ICU probe saved in {}", icu_probe_path)
    # Time measurement
    timer.lap(event_name="Compute ICU probe")

    # Compute condition probe per visit
    logger.info("Computing condition probe per visit...")
    condition_probe_per_visit = ConditionProbe(
        completeness_predictor="per_visit_default"
    )
    condition_probe_per_visit.compute(
        data=data,
        extra_data=AREM_data,
        **ehr_conf["condition"],
    )
    logger.info(
        "Condition probe per visit has shape {}",
        condition_probe_per_visit.predictor.shape,
    )
    condition_probe_per_visit.save(condition_probe_per_visit_path)
    logger.info("Condition per visit probe saved in {}", condition_probe_per_visit_path)
    # Time measurement
    timer.lap(event_name="Compute condition probe per visit")

    # Compute condition probe per condition
    logger.info("Computing condition probe per condition...")
    condition_probe = ConditionProbe(completeness_predictor="per_condition_default")
    condition_probe.compute(
        data=data,
        extra_data=AREM_data,
        **ehr_conf["condition"],
    )
    logger.info(
        "Condition probe per condition has shape {}",
        condition_probe.predictor.shape,
    )
    condition_probe.save(condition_probe_path)
    logger.info("Condition per condition probe saved in {}", condition_probe_path)
    # Time measurement
    timer.lap(event_name="Compute condition probe per condition")

    # Compute note per visit probe
    logger.info("Computing note probe per visit...")
    note_probe_per_visit = NoteProbe(completeness_predictor="per_visit_default")
    note_probe_per_visit.compute(
        data=data,
        extra_data=prod_data,
        **ehr_conf["note"],
    )
    logger.info(
        "Note probe per visit has shape {}", note_probe_per_visit.predictor.shape
    )
    note_probe_per_visit.save(note_probe_per_visit_path)
    logger.info("Note probe per visit saved in {}", note_probe_per_visit_path)
    # Time measurement
    timer.lap(event_name="Compute note probe per visit")

    # Compute note per note probe
    logger.info("Computing note probe per note...")
    note_probe = NoteProbe(completeness_predictor="per_note_default")
    note_probe.compute(
        data=data,
        extra_data=prod_data,
        **ehr_conf["note"],
    )
    logger.info("Note probe per note has shape {}", note_probe.predictor.shape)
    note_probe.save(note_probe_path)
    logger.info("Note probe per note saved in {}", note_probe_path)
    # Time measurement
    timer.lap(event_name="Compute note probe per note")

    # Fit Step Function model for visit
    logger.info("Computing visit step function model...")
    visit_model = StepFunction()
    visit_model.fit(probe=visit_probe)
    logger.info("Visit model has shape {}", visit_model.estimates.shape)
    visit_model.save(visit_model_path)
    logger.info("Visit model saved in {}", visit_model_path)
    # Time measurement
    timer.lap(event_name="Fit Step Function model on visit probe")

    # Fit Rectangle Function model for icu
    logger.info("Computing ICU rectangle function model...")
    icu_model = RectangleFunction()
    icu_model.fit(probe=icu_probe)
    logger.info("ICU model has shape {}", icu_model.estimates.shape)
    icu_model.save(icu_model_path)
    logger.info("ICU model saved in {}", icu_model_path)
    # Time measurement
    timer.lap(event_name="Fit Rectangle Function model on ICU probe")

    # Fit Step Function model for condition per visit
    logger.info("Computing condition step function per visit model...")
    condition_per_visit_model = StepFunction()
    condition_per_visit_model.fit(probe=condition_probe_per_visit)
    logger.info(
        "Condition per visit model has shape {}",
        condition_per_visit_model.estimates.shape,
    )
    condition_per_visit_model.save(condition_per_visit_model_path)
    logger.info("Condition per visit model saved in {}", condition_per_visit_model_path)
    # Time measurement
    timer.lap(event_name="Fit Step Function model on condition probe per visit")

    # Fit Step Function model for condition per condition
    logger.info("Computing condition step function per condition model...")
    condition_model = StepFunction()
    condition_model.fit(probe=condition_probe)
    logger.info(
        "Condition per condition model has shape {}",
        condition_model.estimates.shape,
    )
    condition_model.save(condition_model_path)
    logger.info("Condition per condition model saved in {}", condition_model_path)
    # Time measurement
    timer.lap(event_name="Fit Step Function model on condition probe per condition")

    # Fit Step Function model for note per visit
    logger.info("Computing note step function per visit model...")
    note_per_visit_model = StepFunction()
    note_per_visit_model.fit(probe=note_probe_per_visit)
    logger.info(
        "Note per visit model has shape {}", note_per_visit_model.estimates.shape
    )
    note_per_visit_model.save(note_per_visit_model_path)
    logger.info("Note per visit model saved in {}", note_per_visit_model_path)
    # Time measurement
    timer.lap(event_name="Fit Step Function model on note probe per visit")

    # Fit Step Function model for note per note
    logger.info("Computing note step function per note model...")
    note_model = StepFunction()
    note_model.fit(probe=note_probe)
    logger.info("Note per note model has shape {}", note_model.estimates.shape)
    note_model.save(note_model_path)
    logger.info("Note per note model saved in {}", note_model_path)
    # Time measurement
    timer.lap(event_name="Fit Step Function model on note probe per note")
    timer.stop(script_name="ehr_modeling")

    print("EHR models have been computed and saved ! :sunglasses:")


if __name__ == "__main__":
    app.run()
