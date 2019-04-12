FROM python:3.7

WORKDIR /home

COPY . .

RUN pip install pipenv
RUN pipenv install --system

ENTRYPOINT ["/bin/bash"]
