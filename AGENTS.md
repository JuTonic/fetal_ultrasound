# fetal_ultrasound

Бенчмарк архитектур для классификации плоскостей УЗИ плода (курс NeuroML 2026).
Данные: FETAL_PLANES_DB (12 400 изображений, 1792 пациента), 9 классов: 5 плоскостей + Fetal brain,
разбитый на TT / TC / TV / other. Подробности по данным и решениям — `docs/preprocessing.md`.

## Структура
- `data/raw/` — датасет, не в git, не изменять.
- `data/processed/metadata.csv` — метки, дубликаты, разбиения.
- `fetal/` — `data.py`, `splits.py`, `transforms.py`, `datasets.py`, `prepare.py`.

## Команды
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
python -m pytest -q
python -m fetal.prepare            # не перезаписывает metadata.csv без --force
```

## Факты, которые нельзя нарушать
- Делить по `patient_group`, не по `patient`: 5 пар пациентов продублированы под разными ID.
- `split_grouped` — основное разбиение; `split_official` — официальное + val; `split_naive` —
  image-level с утечкой, только как контроль. В первых двух утечка по пациентам должна быть 0.
- Выбор моделей и гиперпараметров — только по val, не по test.
- Метрики: macro-F1, balanced accuracy, per-class F1.
- Пайплайн изображения: grayscale → letterbox 224 → аугментации (только train) → 3 канала →
  ImageNet mean/std.
- `split_grouped` зависит от версии scikit-learn (поведение `StratifiedGroupKFold` изменилось в 1.8).