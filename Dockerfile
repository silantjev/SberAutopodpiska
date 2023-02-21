FROM python:3.10

WORKDIR /code

RUN pip install --upgrade pip

COPY ./requirements_local_api.txt ./
RUN pip install -r requirements_local_api.txt

COPY dill_pipe.pkl local_api.py ./

EXPOSE 8000

CMD uvicorn local_api:app --host 0.0.0.0 --port 8000
