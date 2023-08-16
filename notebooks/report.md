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
import json
import os
import pandas as pd
from confection import Config
from datetime import datetime
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from cse_210033 import BASE_DIR
from cse_210033.viz import (
    cohort_summary_table,
    ehr_summary_table,
    stats_analysis_summary_tables,
)
```

# CSE210033 - Adjusting for the progressive digitization of health records: working examples on a multi-hospital clinical data warehouse


This study stands on the shoulder of [EDS-TeVa](https://github.com/aphp/edsteva), a python library which which provides a set of tools that aims at modeling the adoption over time and across space of the Electronic Health Records.
#### Library versions:

```python
import cse_210033
import edsteva

print("\033[0m- Study code (CSE-210033): " + "\033[1mv" + cse_210033.__version__)
print("\033[0m- EHR modeling code (EDS-TeVa): " + "\033[1mv" + edsteva.__version__)
```

## I. EHR Modeling


## I.1 Exectution date

```python
with open(
    BASE_DIR
    / "logs"
    / "ehr_modeling"
    / sorted(os.listdir(BASE_DIR / "logs" / "ehr_modeling"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("Code has been exectued the " + "\033[1m" + elapsed_time["timestamp"])
```

## I.2 Execution time

```python
with open(
    BASE_DIR
    / "logs"
    / "ehr_modeling"
    / sorted(os.listdir(BASE_DIR / "logs" / "ehr_modeling"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("\033[1m" + "TOTAL" + ": " + "\033[0m" + elapsed_time["total_elapsed_time"])
    print("\033[1m" + "DETAIL: ")
    for event, time in elapsed_time["event_elapsed_times"].items():
        print("\033[1m" + event + ": " + "\033[0m" + time)
```

## I.3 Config

```python
config_path = (
    BASE_DIR
    / "logs"
    / "ehr_modeling"
    / sorted(os.listdir(BASE_DIR / "logs" / "ehr_modeling"))[-1]
    / "config.cfg"
)
ehr_config = Config().from_disk(config_path, interpolate=True)["ehr_modeling"]
ehr_config
```

## I.4 Result

```python
ehr_summary_table()
```

## II. Cohort selection


## II.1 Exectution date

```python
with open(
    BASE_DIR
    / "logs"
    / "cohort_selection"
    / sorted(os.listdir(BASE_DIR / "logs" / "cohort_selection"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("Code has been exectued the " + "\033[1m" + elapsed_time["timestamp"])
```

## II.2 Execution time

```python
with open(
    BASE_DIR
    / "logs"
    / "cohort_selection"
    / sorted(os.listdir(BASE_DIR / "logs" / "cohort_selection"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("\033[1m" + "TOTAL" + ": " + "\033[0m" + elapsed_time["total_elapsed_time"])
    print("\033[1m" + "DETAIL: ")
    for event, time in elapsed_time["event_elapsed_times"].items():
        print("\033[1m" + event + ": " + "\033[0m" + time)
```

## II.3 Config

```python
config_path = (
    BASE_DIR
    / "logs"
    / "cohort_selection"
    / sorted(os.listdir(BASE_DIR / "logs" / "cohort_selection"))[-1]
    / "config.cfg"
)
cohort_selection_config = Config().from_disk(config_path, interpolate=True)[
    "cohort_selection"
]
cohort_selection_config
```

## II.4 Result

```python
with open(
    BASE_DIR
    / "logs"
    / "cohort_selection"
    / sorted(os.listdir(BASE_DIR / "logs" / "cohort_selection"))[-1]
    / "eds_count.json",
    "r",
) as f:
    eds_count = json.load(f)
    for key, value in eds_count.items():
        print("\033[1m" + str(key) + ": " + "\033[0m" + f"{value:,}".replace(",", " "))
```

```python
cohort_summary_table()
```

## III. Statistical Analysis


## III.1 Exectution date

```python
with open(
    BASE_DIR
    / "logs"
    / "statistical_analysis"
    / sorted(os.listdir(BASE_DIR / "logs" / "statistical_analysis"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("Code has been exectued the " + "\033[1m" + elapsed_time["timestamp"])
```

## III.2 Execution time

```python
with open(
    BASE_DIR
    / "logs"
    / "statistical_analysis"
    / sorted(os.listdir(BASE_DIR / "logs" / "statistical_analysis"))[-1]
    / "timer.json",
    "r",
) as f:
    elapsed_time = json.load(f)
    print("\033[1m" + "TOTAL" + ": " + "\033[0m" + elapsed_time["total_elapsed_time"])
    print("\033[1m" + "DETAIL: ")
    for event, time in elapsed_time["event_elapsed_times"].items():
        print("\033[1m" + event + ": " + "\033[0m" + time)
```

## III.3 Config

```python
with open(
    BASE_DIR
    / "logs"
    / "statistical_analysis"
    / sorted(os.listdir(BASE_DIR / "logs" / "statistical_analysis"))[-1]
    / "config.json",
    "r",
) as f:
    stats_config = json.load(f)
stats_config
```

## III.4 Result

```python
summary_tables = stats_analysis_summary_tables(stats_config)
```

```python

```
