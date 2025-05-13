import logging
import random
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import click
import numpy as np
import yaml
from prometheus_client import Gauge, Histogram, start_http_server
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


class ConfigReloader(FileSystemEventHandler):
    def __init__(self, config_path: Path, callback: Callable):
        super().__init__()
        self.callback = callback
        self.config_path = config_path

    def on_modified(self, event: FileModifiedEvent) -> Any:
        if Path(event.src_path) == self.config_path:
            self.callback()


class MetricGenerator:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.active_transitions: dict[str, dict] = {}
        self.metrics: dict[str, Any] = {}
        self.lock = Lock()
        self.last_update = time.time()
        self.load_config()

    def load_config(self) -> None:
        logger.info("Load config from file %s", self.config_path)
        with open(self.config_path) as fd:
            config = yaml.safe_load(fd)

        self.transition_time = config.get("transition_time", int(timedelta(minutes=5).total_seconds()))

        with self.lock:
            for name in self.metrics:
                if not any(m["name"] == name for m in config["metrics"]):
                    logger.info("Metric %s deleted", name)
                    del self.metrics[name]

            for metric_def in config["metrics"]:
                name = metric_def["name"]
                if name not in self.metrics:
                    self.create_metric(metric_def)
                else:
                    self.schedule_transition(name, metric_def)

    def schedule_transition(self, metric_name: str, new_def: dict):
        now = time.time()
        old_params = self.get_interpolated_params(metric_name)
        new_params = new_def["profile"]

        # Запоминаем начальные и целевые значения
        self.active_transitions[metric_name] = {
            "start_time": now,
            "end_time": now + self.transition_time,
            "start_params": old_params,
            "target_params": new_params,
        }

    def get_interpolated_params(self, metric_name: str) -> dict:
        transition = self.active_transitions.get(metric_name)
        if not transition:
            return self.metrics[metric_name]["profile"]

        now = time.time()
        if now >= transition["end_time"]:
            self.metrics[metric_name]["profile"] = transition["target_params"]
            del self.active_transitions[metric_name]
            return transition["target_params"]

        # Линейная интерполяция
        progress = (now - transition["start_time"]) / self.transition_time
        interpolated = {}

        for param in ["amplitude", "period", "base", "rate"]:
            if param in transition["start_params"]:
                start_val = transition["start_params"][param]
                target_val = transition["target_params"][param]
                interpolated[param] = start_val + (target_val - start_val) * progress

        return {**transition["start_params"], **interpolated}

    def create_metric(self, metric_def: dict) -> None:
        logger.info("Create metric %s", metric_def["name"])
        if metric_def["type"] == "gauge":
            self.metrics[metric_def["name"]] = {
                "type": "gauge",
                "name": metric_def["name"],
                "obj": Gauge(metric_def["name"], metric_def["help"]),
                "profile": metric_def["profile"],
                "noise": metric_def.get("noise"),
                "bounds": metric_def.get("bounds"),
                "state": {},
            }
        elif metric_def["type"] == "histogram":
            self.metrics[metric_def["name"]] = {
                "type": "histogram",
                "name": metric_def["name"],
                "obj": Histogram(
                    metric_def["name"],
                    metric_def["help"],
                    buckets=metric_def.get("buckets", []),
                ),
                "profile": metric_def["profile"],
                "state": {"value": metric_def["profile"].get("start", 0)},
            }

    def generate_value(self, metric: dict) -> float:
        profile = self.get_interpolated_params(metric["name"])
        profile_type = profile["type"]
        t = time.time()

        if profile_type == "periodic":
            period = profile["period"]
            amplitude = profile["amplitude"]
            phase = profile.get("phase", 0)
            base = profile.get("base", 0)
            return base + amplitude * np.sin(2 * np.pi * (t - phase) / period)

        elif profile_type == "sawtooth":
            rate = profile["rate"]
            reset_interval = profile["reset_interval"]
            cycles = t // reset_interval
            return rate * (t - cycles * reset_interval)

        elif profile_type == "constant":
            return profile.get("value", 0)

        else:
            return 0

    def add_noise(self, value: float, noise_def: dict) -> float:
        if not noise_def:
            return value

        if noise_def["type"] == "uniform":
            return value + random.uniform(-noise_def["magnitude"], noise_def["magnitude"])

        elif noise_def["type"] == "percentage":
            return value * (1 + random.uniform(-noise_def["magnitude"] / 100, noise_def["magnitude"] / 100))
        return value

    def apply_bounds(self, value: float, bounds: dict) -> float:
        if bounds:
            return max(bounds.get("min", -np.inf), min(value, bounds.get("max", np.inf)))
        return value

    def update_metrics(self):
        with self.lock:
            for name, metric in self.metrics.items():
                raw_value = self.generate_value(metric)

                noised_value = self.add_noise(raw_value, metric.get("noise", {}))

                final_value = self.apply_bounds(noised_value, metric.get("bounds", {}))

                if metric["type"] == "gauge":
                    metric["obj"].set(final_value)
                elif metric["type"] == "histogram":
                    metric["obj"].observe(final_value)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--port", type=int, default=8041)
def main(config_path: Path, port: int):
    start_http_server(port)
    generator = MetricGenerator(config_path)

    event_handler = ConfigReloader(config_path, generator.load_config)
    observer = Observer()
    observer.schedule(event_handler, path=config_path.parent, recursive=False)
    observer.start()

    try:
        while True:
            generator.update_metrics()
            time.sleep(15)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
