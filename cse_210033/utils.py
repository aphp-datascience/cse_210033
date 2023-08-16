import datetime
import json
import os
import time
from datetime import timedelta

from loguru import logger

from cse_210033 import BASE_DIR


def dump_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


class timemeasure:
    def __init__(self):
        self.t_start = time.time()
        self.t1 = time.time()
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.event_elapsed_times = {}
        self.step_number = 0

    def lap(self, event_name: str):
        t2 = time.time()
        elapsed_time = str(timedelta(seconds=t2 - self.t1))
        self.t1 = time.time()
        self.event_elapsed_times[
            "{} - {}".format(self.step_number, event_name)
        ] = elapsed_time
        self.step_number += 1
        logger.info(f"{event_name} took {elapsed_time} seconds")

    def stop(self, script_name: str, create_folder: bool = False):
        t_end = time.time()
        total_elapsed_time = str(timedelta(seconds=t_end - self.t_start))

        parent_dir = BASE_DIR / "logs" / script_name
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
        if create_folder:
            path_dir = parent_dir / self.timestamp.strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(path_dir)
        else:
            path_dir = (
                parent_dir / sorted(os.listdir(BASE_DIR / "logs" / script_name))[-1]
            )

        logs = {
            "total_elapsed_time": total_elapsed_time,
            "timestamp": self.timestamp.strftime("%d %B, %Y at %H:%M:%S"),
            "event_elapsed_times": self.event_elapsed_times,
            "script_name": script_name,
        }
        dump_data(
            logs,
            path=path_dir / "timer.json",
        )
        logger.info(f"{script_name} took {total_elapsed_time} seconds")
