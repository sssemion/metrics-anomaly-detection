# Synthetic load generator

Генератор синтетических метрик для тестирования.

## Features

- Поддержка типов метрик: Gauge, Histogram
- Настраиваемые профили генерации данных
- Добавление шума к значениям
- Hot-reload по обновлению конфига

## Конфигурация

Создайте `config.yaml`:

```yaml
metrics:
  - name: network_latency
    type: gauge
    help: "Simulated network latency in ms"
    labels: ["node"]
    profile:
      type: periodic
      amplitude: 50    # ±50 ms
      period: 60       # 1 minute cycle
      base: 100        # base value
    noise:
      type: uniform
      magnitude: 5     # ±5 ms
    bounds:
      min: 50
      max: 200
```

### Параметры метрик

| Параметр    | Обязательный | Описание                          |
|-------------|--------------|-----------------------------------|
| `name`      | Да           | Имя метрики (Prometheus format)   |
| `type`      | Да           | Тип: gauge, counter, histogram    |
| `help`      | Да           | Описание метрики                  |
| `labels`    | Нет          | Список меток                      |
| `profile`   | Да           | Профиль генерации значений        |
| `noise`     | Нет          | Параметры шума                    |
| `bounds`    | Нет          | Ограничения значений              |
| `buckets`   | Для histogram| Границы бакетов                   |

## Профили генерации

### 1. Периодический (`periodic`)

```math
f(t) = \text{base} + \text{amplitude} \cdot \sin\left(\frac{2\pi(t - \text{phase})}{\text{period}}\right)
```

где:
- `amplitude` - амплитуда колебаний
- `period` - период в секундах
- `phase` - начальная фаза (по умолчанию 0)
- `base` - базовое значение

### 2. Пилообразный (`sawtooth`)

Линейный рост со сбросом:
```math
f(t) = \text{rate} \cdot \left(t \mod \text{reset\_interval}\right)
```

Параметры:
- `rate` - скорость роста
- `reset_interval` - интервал сброса (сек)

### 3. Постоянный (`constant`)

```math
f(t) = \text{value}
```

Параметр:
- `value` - фиксированное значение

## Типы шума

### Равномерный (`uniform`)

```math
f(x) = x + \mathcal{U}(-\text{magnitude}, +\text{magnitude})
```

Параметры:
- `magnitude` - амплитуда шума

### Процентный (`percentage`)

```math
f(x) = x \cdot \left(1 + \mathcal{U}\left(-\frac{\text{magnitude}}{100}, +\frac{\text{magnitude}}{100}\right)\right)
```

Параметры:
- `magnitude` - процент отклонения

## Запуск

```bash
uv run main.py config.yaml [--port 0000]
```

Метрики доступны на порту 8041 (по умолчанию):
```bash
curl http://localhost:8041/metrics
```

## Динамическое обновление

1. Измените `config.yaml`
2. Сохраните файл
3. Приложение автоматически применит изменения

## Интеграция с Prometheus

Добавьте в `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'load_generator'
    static_configs:
      - targets: ['localhost:8041']
```
