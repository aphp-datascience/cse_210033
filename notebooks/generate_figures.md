---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: teva_study_client
    language: python
    name: teva_study_client
---

```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
```

```python
import sys
import warnings

import pandas as pd
import typer
from confection import Config
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from edsteva.probes import ConditionProbe, NoteProbe, VisitProbe
from loguru import logger
from rich import print

from cse_210033 import BASE_DIR
from cse_210033.viz import (
    plot_ehr_context,
    plot_ehr_models,
    plot_ehr_rectangle_model,
    plot_epidemiology_indicators,
    plot_model_goodness_fit,
    plot_quality_indicators,
    plot_sensibility_care_site,
)

warnings.filterwarnings("ignore")
```

```python
config_path = BASE_DIR / "conf" / "config.cfg"
config = Config().from_disk(config_path, interpolate=True)
if config["debug"]["debug"]:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
```

## Load Probes

```python
ehr_modeling_folder_path = BASE_DIR / "data" / "ehr_modeling"
probes_folder_path = ehr_modeling_folder_path / "probes"

visit_probe_path = probes_folder_path / "visit.pickle"
condition_probe_path = probes_folder_path / "condition.pickle"
condition_probe_per_visit_path = probes_folder_path / "condition_per_visit.pickle"
note_probe_path = probes_folder_path / "note.pickle"
note_probe_per_visit_path = probes_folder_path / "note_per_visit.pickle"
icu_probe_path = probes_folder_path / "icu_rectangle.pickle"

visit_probe = VisitProbe()
visit_probe.load(visit_probe_path)

condition_probe = ConditionProbe()
condition_probe.load(condition_probe_path)
condition_probe_per_visit = ConditionProbe()
condition_probe_per_visit.load(condition_probe_per_visit_path)

note_probe = NoteProbe()
note_probe.load(note_probe_path)
note_probe_per_visit = NoteProbe()
note_probe_per_visit.load(note_probe_per_visit_path)

icu_probe = VisitProbe()
icu_probe.load(icu_probe_path)
```

## Load Models

```python
models_folder_path = ehr_modeling_folder_path / "models"
visit_model_path = models_folder_path / "visit.pickle"
condition_model_path = models_folder_path / "condition.pickle"
condition_per_visit_model_path = models_folder_path / "condition_per_visit.pickle"
note_model_path = models_folder_path / "note.pickle"
note_per_visit_model_path = models_folder_path / "note_per_visit.pickle"
icu_model_path = models_folder_path / "icu_rectangle.pickle"

visit_model = StepFunction()
visit_model.load(visit_model_path)

condition_model = StepFunction()
condition_model.load(condition_model_path)
condition_per_visit_model = StepFunction()
condition_per_visit_model.load(condition_per_visit_model_path)

note_model = StepFunction()
note_model.load(note_model_path)
note_per_visit_model = StepFunction()
note_per_visit_model.load(note_per_visit_model_path)

icu_model = RectangleFunction()
icu_model.load(icu_model_path)
```

## Load post_processed data

```python
statistical_analysis_path = BASE_DIR / "data" / "statistical_analysis"
cs_count_summary = pd.read_pickle(statistical_analysis_path / "cs_count_summary.pkl")
quality_indicators = {}
icu_quality_indicators = {}
epidemiology_indicators = {}
quality_events = [
    "hospit_visit",
    "emergency_visit",
    "consultation_note",
    "prescription_note",
]
icu_quality_events = [
    "icu_visit",
    "icu_visit_rectangle",
]
epidemiology_events = [
    "bronchiolitis_condition",
    "flu_condition",
]
for quality_event in quality_events:
    quality_indicators[quality_event] = pd.read_pickle(
        statistical_analysis_path / "{}.pkl".format(quality_event)
    )
for icu_quality_event in icu_quality_events:
    icu_quality_indicators[icu_quality_event] = pd.read_pickle(
        statistical_analysis_path / "{}.pkl".format(icu_quality_event)
    )
for epidemiology_event in epidemiology_events:
    epidemiology_indicators[epidemiology_event] = pd.read_pickle(
        statistical_analysis_path / "{}.pkl".format(epidemiology_event)
    )
```

## Figure 1: EHR context

```python
ehr_context_chart, ehr_context_data = plot_ehr_context(
    visit_predictor=visit_probe.predictor,
    condition_predictor=condition_probe.predictor,
    note_predictor=note_probe.predictor,
    start_date=config["cohort_selection"]["start_date"],
)
ehr_context_chart.save(BASE_DIR / "figures" / "figure_1" / "ehr_context_chart.html")
ehr_context_data.to_csv(BASE_DIR / "figures" / "figure_1" / "ehr_context_data.csv")
display(ehr_context_chart)
print("Figure 1: EHR context have been saved")
```

## Figure 2: Good example of fitted ehr models

```python
ehr_model_chart, ehr_models_data = plot_ehr_models(
    visit_probe=visit_probe,
    note_probe=note_probe,
    note_probe_per_visit=note_probe_per_visit,
    condition_probe_per_visit=condition_probe_per_visit,
    visit_model=visit_model,
    note_model=note_model,
    note_per_visit_model=note_per_visit_model,
    condition_per_visit_model=condition_per_visit_model,
    config=config,
)
ehr_model_chart.save(BASE_DIR / "figures" / "figure_2" / "ehr_model_chart.html")
ehr_models_data.to_csv(BASE_DIR / "figures" / "figure_2" / "ehr_models_data.csv")
display(ehr_model_chart)
print("Figure 2: Good example of fitted ehr models have been saved")
```

# Figure 3: Quality indicators

```python
quality_charts, quality_data = plot_quality_indicators(
    quality_indicators=quality_indicators,
    cs_count_summary=cs_count_summary,
    config=config,
)
quality_data.to_csv(BASE_DIR / "figures" / "figure_3" / "quality_indicators.csv")
quality_charts.save(BASE_DIR / "figures" / "figure_3" / "quality_indicators.html")
display(quality_charts)
print("Figure 3: Quality Indicator charts has been saved")
```

# Figure 4: Sensibility analysis for care site

```python
sens_cs_chart, sens_cs_data = plot_sensibility_care_site(quality_indicators, config)
sens_cs_chart.save(BASE_DIR / "figures" / "figure_4" / "sens_cs_chart.html")
sens_cs_data.to_csv(BASE_DIR / "figures" / "figure_4" / "sens_cs_data.csv")
display(sens_cs_chart)
print("Figure 4: Sensibility analysis for care site has been saved")
```

# Figure 5: Epidemiology indicators

```python
epidemiology_charts, epidemiology_data = plot_epidemiology_indicators(
    epidemiology_indicators=epidemiology_indicators,
    cs_count_summary=cs_count_summary,
    config=config,
)
epidemiology_data.to_csv(
    BASE_DIR / "figures" / "figure_5" / "epidemiology_indicators.csv"
)
epidemiology_charts.save(
    BASE_DIR / "figures" / "figure_5" / "epidemiology_indicators.html"
)
display(epidemiology_charts)
print("Figure 5: Epidemiology indicators charts have been saved")
```

# eFigure 1: Modeling of EHR adoption for ICUs records using rectangular functions

```python
ehr_rectangle_model_chart, ehr_rectangle_models_data = plot_ehr_rectangle_model(
    visit_probe=visit_probe,
    icu_probe=icu_probe,
    visit_model=visit_model,
    icu_model=icu_model,
    config=config,
)
ehr_rectangle_model_chart.save(
    BASE_DIR / "figures" / "efigure_1" / "ehr_rectangle_model_chart.html"
)
ehr_rectangle_models_data.to_csv(
    BASE_DIR / "figures" / "efigure_1" / "ehr_rectangle_models_data.csv"
)
display(ehr_rectangle_model_chart)
print(
    "eFigure 1: Modeling of EHR adoption for ICUs records using rectangular functions"
)
```

# eFigure2: Goodness-of-fit of the step-function modeling

```python
model_goodness_fit_chart, model_goodness_fit_data = plot_model_goodness_fit(
    visit_probe=visit_probe,
    note_probe=note_probe,
    note_probe_per_visit=note_probe_per_visit,
    condition_probe_per_visit=condition_probe_per_visit,
    visit_model=visit_model,
    note_model=note_model,
    note_per_visit_model=note_per_visit_model,
    condition_per_visit_model=condition_per_visit_model,
    config=config,
)
model_goodness_fit_chart.save(
    BASE_DIR / "figures" / "efigure_2" / "model_goodness_fit_chart.html"
)
model_goodness_fit_data.to_csv(
    BASE_DIR / "figures" / "efigure_2" / "model_goodness_fit_data.csv"
)
display(model_goodness_fit_chart)
print("eFigure2: Goodness-of-fit of the step-function modeling has been saved")
```

# eFigure3: ICU Quality indicators

```python
icu_quality_charts, icu_quality_data = plot_quality_indicators(
    quality_indicators=icu_quality_indicators,
    cs_count_summary=cs_count_summary,
    icu_box_position=True,
    config=config,
)
icu_quality_data.to_csv(
    BASE_DIR / "figures" / "efigure_3" / "icu_quality_indicators.csv"
)
icu_quality_charts.save(
    BASE_DIR / "figures" / "efigure_3" / "icu_quality_indicators.html"
)
display(icu_quality_charts)
print("eFigure 3: ICU Quality Indicator charts has been saved")
```

# eFigure 4: Sensibility analysis for care site ICU

```python
sens_cs_icu_chart, sens_cs_icu_data = plot_sensibility_care_site(
    icu_quality_indicators, config
)
sens_cs_icu_chart.save(BASE_DIR / "figures" / "efigure_4" / "sens_cs_icu_chart.html")
sens_cs_icu_data.to_csv(BASE_DIR / "figures" / "efigure_4" / "sens_cs_icu_data.csv")
display(sens_cs_icu_chart)
print("eFigure 4: Sensibility analysis for Care site has been saved")
```

```python

```
