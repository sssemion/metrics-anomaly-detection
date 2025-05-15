import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import TypedDict

import aiohttp
import click
import numpy as np
import yaml
from prometheus_client import Gauge, start_http_server
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)


class InvalidConfig(Exception):
    pass


class Metric(TypedDict):
    labels: dict[str, str]
    values: list[float]


class AsyncPrometheusClient:
    def __init__(self, url: str):
        self.url = url

    async def fetch_metric(
        self, query: str, start: datetime, end: datetime, step: str
    ) -> list[Metric]:
        params = {
            "query": query,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": step,
        }

        result: list[Metric] = []
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(
                f"{self.url}/api/v1/query_range", params=params
            ) as response:
                data = await response.json()
                for raw_metric in data["data"]["result"]:
                    raw_metric["metric"].pop("__name__", None)
                    metric = Metric(
                        labels=raw_metric["metric"],
                        values=[item[1] for item in raw_metric["values"][-32:]],
                    )
                    result.append(metric)
        return result


class AnomalyDetector:
    detector_start: int = 0

    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.metric_cache: dict[tuple[str, tuple], Gauge] = {}
        start_http_server(self.config["exporter_port"])

        self.values_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20000))
        self.mse_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20000))

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            config = yaml.safe_load(f)

        # Валидация конфигурации
        required_fields = {
            "prometheus_url",
            "exporter_port",
            "window_size",
            "scrape_interval",
            "queries",
        }
        if not required_fields.issubset(config):
            raise InvalidConfig("Missing required config fields")

        return config

    def _load_model(self) -> Model:
        return load_model(self.config["model_path"])

    def _get_gauge(self, query_name: str, labels: tuple[str, ...]) -> Gauge:
        """Динамически создает метрику при первом появлении комбинации лейблов"""
        # Создаем метрику только если ее еще нет
        if (query_name, labels) not in self.metric_cache:
            metric = Gauge(
                f"anomaly_detector__{query_name}",
                "Auto-generated anomaly_score metric",
                labels,
            )
            self.metric_cache[(query_name, labels)] = metric

        return self.metric_cache[(query_name, labels)]

    @cached_property
    def client(self) -> AsyncPrometheusClient:
        return AsyncPrometheusClient(self.config["prometheus_url"])

    async def fetch_data(self) -> list[list[Metric]]:
        fetch_tasks = []
        logger.info("Fetch %d queries", len(self.config["queries"]))
        for query in self.config["queries"]:
            window = timedelta(**query["window"])
            end = datetime.now(tz=timezone.utc)
            start = end - window
            step = window.total_seconds() / self.config["window_size"]

            fetch_tasks.append(
                self.client.fetch_metric(query["query"], start, end, step)
            )
        return await asyncio.gather(*fetch_tasks)

    def _process_metrics(self, metrics: list[list[Metric]]):
        tensors = []
        for q, q_metrics in zip(self.config["queries"], metrics, strict=True):
            name = q["name"]
            history_empty = name not in self.values_history
            for q_metric in q_metrics:
                if history_empty:
                    self.values_history[name].extend(q_metric["values"])
                else:
                    self.values_history[name].append(q_metric["values"][-1])
            scaler = StandardScaler()
            scaler.fit(np.array(self.values_history[name]).reshape(-1, 1))
            tensors.extend(
                scaler.transform(np.array(q_metric["values"]).reshape(-1, 1))
                for q_metric in q_metrics
            )
        input_data = np.array(tensors)
        preds = self.model.predict(input_data)
        mse = np.mean(np.square(input_data - preds), axis=(1, 2))
        imse = iter(mse)
        for q, q_metrics in zip(self.config["queries"], metrics, strict=True):
            name = q["name"]
            for q_metric in q_metrics:
                metric_mse = next(imse)
                self.mse_history[name].append(metric_mse)
                q999 = np.quantile(self.mse_history[name], 0.999)
                if metric_mse >= q999:
                    anomaly_score = 1
                else:
                    anomaly_score = metric_mse / (q999 + 1e-6)

                if q.get("binarize", False):
                    threshold = q.get("threshold", 0.9)
                    if anomaly_score >= threshold:
                        anomaly_score = 1
                    else:
                        anomaly_score = 0

                if time.time() - self.detector_start < self.config.get("initial_offset", 300):
                    logger.info("Waiting for initial offset to pass, set anomaly_score to -1")
                    anomaly_score = -1

                gauge = self._get_gauge(name, tuple(q_metric["labels"].keys()))
                gauge.labels(**q_metric["labels"]).set(anomaly_score)

    def run(self):
        """Основной цикл сбора данных"""
        self.detector_start = time.time()
        while True:
            logger.info("Start processing iteration")
            start_time = time.time()

            metrics = asyncio.run(self.fetch_data())
            self._process_metrics(metrics)

            # Контроль частоты опроса
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config["scrape_interval"] - elapsed)
            logger.info("Sleep for %ds", sleep_time)
            time.sleep(sleep_time)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def main(config_path: Path) -> None:
    detector = AnomalyDetector(config_path)
    detector.run()


if __name__ == "__main__":
    main()
