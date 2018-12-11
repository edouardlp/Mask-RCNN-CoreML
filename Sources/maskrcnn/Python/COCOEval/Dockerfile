FROM python:3.6
WORKDIR /usr/src/app
RUN pip install Cython
RUN pip install numpy
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT [ "python", "./task.py" ]