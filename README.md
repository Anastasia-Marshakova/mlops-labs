# Лабораторная работа №1. Воспроизводимый проект: данные и пайплайн

## Автор и вариант

GitHub: Anastasia-Marshakova.
Вариант 10 — Covertype (UCI).

## Цель работы

Создать воспроизводимый ML-проект с зафиксированным окружением,
версионированием данных и пайплайном обработки.

## Данные и задача

Источник: UCI Covertype.
Загрузка выполняется через sklearn.datasets.fetch_covtype.

Задача — многоклассовая классификация типа лесного покрытия.
Целевая переменная — Cover_Type, возможные значения от 1 до 7.

Для обучения выбрана первая лесная зона:
Wilderness_Area_0 = 1 в именах колонок scikit-learn.
Остальные зоны зарезервированы для потока данных в ЛР4.

Версии данных:
- data-v1: 10 000 объектов первой зоны;
- data-v2: 20 000 объектов той же зоны, включая все объекты v1.

Полный загруженный набор находится в data/external/ вне Git и DVC.
Рабочий срез data/raw/covertype.csv хранится через DVC.

## Структура проекта

- src/ — модули обработки, обучения и оценки;
- notebooks/ — рабочий ноутбук;
- tests/ — каталог для тестов;
- data/raw/ — исходный рабочий срез под контролем DVC;
- data/processed/ — обучающая и тестовая выборки;
- data/features/ — подготовленные признаки;
- models/ — обученная модель;
- reports/metrics.json — метрики;
- params.yaml — параметры;
- dvc.yaml и dvc.lock — описание и состояние пайплайна;
- pyproject.toml и uv.lock — окружение.

## Развёртывание в Google Colab

Создать секреты GITHUB_USER, GITHUB_TOKEN, DAGSHUB_USER,
DAGSHUB_TOKEN и разрешить доступ к ним из ноутбука.
Токены не записывать в код и не выводить на экран.

Клонировать публичный репозиторий:

```python
!git clone https://github.com/Anastasia-Marshakova/mlops-labs.git
%cd /content/mlops-labs
```

Установить окружение из сохранённого lock-файла:

```python
!pip install -q uv
!uv export --frozen --no-hashes --group dev -o /content/requirements.lock.txt
!pip install -q -r /content/requirements.lock.txt
```

При воспроизведении uv.lock не пересоздаётся.

## Доступ к хранилищу DVC

Адрес хранилища сохранён в .dvc/config.
В новом клоне необходимо задать локальные ключи:

```python
import subprocess
from google.colab import userdata

token = userdata.get("DAGSHUB_TOKEN")

for key in ("access_key_id", "secret_access_key"):
    result = subprocess.run(
        ["dvc", "remote", "modify", "--local", "origin", key, token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Не удалось настроить локальные ключи DVC")
```

Ключи сохраняются только в .dvc/config.local, исключённом из Git.

## Воспроизведение

```python
!dvc pull
!dvc repro
!cat reports/metrics.json
```

dvc pull загружает данные и результаты этапов.
dvc repro проверяет зависимости и выполняет только необходимые этапы.
Повторный запуск пропускает неизменившиеся этапы.

## Этапы пайплайна

1. prepare — проверка данных, удаление полных дубликатов,
   стратифицированное разбиение 80/20 с фиксированным seed.
2. featurize — удаление четырёх индикаторов лесной зоны:
   внутри выбранной зоны они постоянны.
3. train — обучение RandomForestClassifier.
4. evaluate — расчёт accuracy и F1 macro на тестовой выборке,
   сохранение reports/metrics.json и проверка порога accuracy.

В наборе нет даты наблюдения, поэтому разбиение по времени не используется.
После подготовки остаётся 50 признаков.

## Результаты

| Версия | Всего строк | Размер теста | Accuracy | F1 macro |
|---|---:|---:|---:|---:|
| data-v1 | 10000 | 2000 | 0.8520 | 0.7305 |
| data-v2 | 20000 | 4000 | 0.8665 | 0.7930 |

На второй версии обе метрики выше. Однако тестовые выборки различаются,
поэтому это сравнение не изолирует влияние размера обучающей выборки.

## Переключение версий

Перед переключением рабочее дерево Git должно быть чистым.

```python
!git checkout data-v1
!dvc checkout
!dvc repro
```

Возврат ко второй версии:

```python
!git checkout data-v2
!dvc checkout
!dvc repro
```

Возврат к основной ветке:

```python
!git checkout main
!dvc checkout
```

Проверено: обе версии восстанавливают соответствующее количество строк
и метрики. dvc repro после переключения пропускает все этапы.

Если нужных объектов нет в локальном кеше, выполнить dvc pull.

## Проверки кода

```python
!pre-commit install
!pre-commit run --all-files
```

Настроены Ruff, форматирование, проверка YAML, поиск приватных ключей
и ограничение размера добавляемых файлов 1 МБ.

## Хранение результатов

Git хранит код, параметры, lock-файлы, метафайлы DVC и метрики.
Хранилище DVC содержит рабочие данные, промежуточные таблицы и модель.

Перед отправкой изменений в GitHub выполняется dvc push.
