# Car Price Prediction

Домашнее задание №1 по машинному обучению: регрессия для предсказания цены автомобилей.

## Структура репозитория

- `HW1_Regression_with_inference_base.ipynb` - ноутбук с полным EDA, обучением и сравнением моделей
- `model_pipeline.pkl` - сохраненная модель Ridge и препроцессоры
- `app.py` - Streamlit-приложение для прогнозирования
- `conclusions.md` - выводы по проделанной работе
- `requirements.txt` - зависимости

## Запуск приложения

```bash
streamlit run app.py
```

## Результаты

Лучшая модель: **Ridge regression** с категориальными признаками
- R² на test: 0.672
- Business metric (прогнозы с ошибкой ≤10%): 24.5%
