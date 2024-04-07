FROM python:3.10-slim-bullseye
 
EXPOSE 8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt
 
RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt
 

COPY ./app.py /app/app.py

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]