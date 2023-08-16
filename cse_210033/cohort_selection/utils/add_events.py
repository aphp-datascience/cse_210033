from datetime import date

from edsteva.utils.typing import DataFrame, Series


def add_patient_info(visit: DataFrame, person: DataFrame):
    # Compute ages
    person["age_today"] = _compute_age_today(
        person.birth_datetime, person.death_datetime
    )
    visit = visit.merge(person, on="person_id", how="inner")
    visit["age_at_stay"] = _compute_age_at_stay(
        birth_dates=visit.birth_datetime, cohort_stay_start=visit.cohort_stay_start
    )
    return visit


def _compute_age_at_stay(birth_dates: Series, cohort_stay_start: Series):
    ages = (
        cohort_stay_start.dt.year
        - birth_dates.dt.year
        - (
            (
                cohort_stay_start.dt.strftime("%m-%d")
                < birth_dates.dt.strftime("%m-%d")
            ).astype(int)
        )
    )
    return ages


def _compute_age_today(birth_dates: Series, death_dates: Series):
    death_ages = (
        death_dates.dt.year
        - birth_dates.dt.year
        - (
            (
                death_dates.dt.strftime("%m-%d") < birth_dates.dt.strftime("%m-%d")
            ).astype(int)
        )
    )
    now = date.today()
    alive_ages = (
        now.year
        - birth_dates.dt.year
        - ((now.strftime("%m-%d") < birth_dates.dt.strftime("%m-%d")).astype(int))
    )
    return death_ages.mask(death_ages.isna(), alive_ages)
