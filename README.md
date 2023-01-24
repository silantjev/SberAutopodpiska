# SberAutopodpiska
Задача: На сайт «СберАвтоподписка» заходят пользователи и совершают некоторые действия (или не совершают). Требуется построить модель, которая по входным значениям будет угадывать, совершит ли пользователь целевое действие при посещении сайта.

Модель должна брать на вход все атрибуты, типа utm_*, device_*, geo_*, и отдавать на выход 0/1 (1 — если пользователь совершит любое целевое действие, 0 в противном случае).


Цель проекта — тренировка построения предсказательной модели.


Источник: Skillbox.ru


Язык: Python 3


Автор: А. В. Силантьев


Файлы:

autopodpiska.ipynb — тетрадь с анализом данных и подбором модели;

pipeline.py строит конвеер с выбранной моделью;

dill_pipe.pkl — полученный конвеер;

local_api.py — локальный api-сервер, работающий на конвеере dill_pipe.pkl;

api_test.ipynb — тетрадь с тестом api-сервера;

data пака для данных.

Данные для проекта — два файла:

ga_sessions.csv и ga_hits.csv,

которые можно скачать по адресу

https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw?usp=sharing

и положить в папку data.

Используемые модули:

os, math, warnings, re, dill=0.3.6, numpy=1.24.0, scipy=1.9.3, pandas=1.5.2, matplotlib=3.6.2, requests=2.28.1, fastapi=0.88.0, pydantic=1.10.2, uvicorn=0.20.0, scikit-learn=1.2.0


Файлы с расширением py следует запускать программой-интерпретатором python3, например:

python3 pipeline.py

или

python pipeline.py

Для запуска файлов с расширением ipynb используется jupyter notebook


Установка модулей:

pip install dill numpy scipy pandas matplotlib requests pydantic scikit-learn "fastapi[all]"
