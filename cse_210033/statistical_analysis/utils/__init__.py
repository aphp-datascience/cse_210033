import catalogue

from cse_210033.statistical_analysis.utils.key_variables import (
    compute_condition_incidence,
    compute_duration_after_event,
    compute_event_during_cohort_stay,
)

key_functions = catalogue.create("cse_210033", "key_functions")

key_functions.register("compute_condition_incidence", func=compute_condition_incidence)
key_functions.register(
    "compute_duration_after_event", func=compute_duration_after_event
)
key_functions.register(
    "compute_event_during_cohort_stay", func=compute_event_during_cohort_stay
)
