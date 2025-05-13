```markdown
# Detector Module

Модуль для детектирования аномалий в метриках Prometheus с использованием ML-модели.

## Features

- Интеграция с Prometheus через API
- Поддержка произвольных PromQL-запросов
- Загрузка предобученных ML-моделей (Keras/HDF5)
- Автоматическая нормализация данных
- Экспорт anomaly scores/binary в Prometheus

## Конфигурация

Создайте `detector.yaml`:
```yaml
prometheus_url: http://localhost:9090
model_path: lstm-ae-32-model-bs256.h5
window_size: 32
exporter_port: 8042
scrape_interval: 30
initial_offset: 600  # 10 минут калибровки

queries:
  - name: cpu_anomaly
    query: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)
    window:
      hours: 1
    binarize: true
    threshold: 0.85
```

### Параметры конфига:
| Параметр         | Описание                                |
|------------------|-----------------------------------------|
| `prometheus_url` | URL сервера Prometheus                  |
| `model_path`     | Путь к файлу ML-модели (.h5)            |
| `window_size`    | Количество точек, принимаемых моделью   |
| `exporter_port`  | Порт для экспорта метрик                |
| `scrape_interval`| Интервал сбора данных (секунды)         |
| `initial_offset` | Период калибровки порогов (секунды)     |

### Параметры запросов:
```yaml
- name: unique_metric_name  # Уникальное имя метрики
  query: promql_query       # Произвольный PromQL-запрос
  window: {hours: 1, minutes: 30}        # Временное окно данных
  binarize: false           # Преобразовать score в 0/1
  threshold: 0.9            # Порог для бинаризации (0-1)
```

## Запуск
```bash
uv run detector.py detector.yaml
```

## Интерпретация результатов
- **Anomaly Score (0-1)**: Вероятность аномалии
- **Binarized (0/1)**: 1 = аномалия обнаружена, 0 = норма

## Калибровка модели
Первые `initial_offset` секунд модуль:
1. Собирает исторические данные
2. Настраивает пороговые значения
3. Калибрует нормализацию данных

В этот период anomaly score равен -1.

## Примеры запросов
### Мониторинг HTTP-ошибок
```yaml
- name: http_errors
  query: rate(http_requests_total{status=~"5.."})
  window:
    minutes: 30
  binarize: true
  threshold: 0.8
```

### Аномалии использования памяти
```yaml
- name: memory_usage
  query: node_memory_MemFree_bytes / node_memory_MemTotal_bytes
  window:
    hours: 2
  binarize: false
```

## Интеграция с Prometheus

Добавьте в `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'anomaly_detector'
    static_configs:
      - targets: ['localhost:8042']
```
