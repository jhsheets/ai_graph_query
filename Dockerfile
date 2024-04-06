FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git

RUN pip install langchain \
	&& pip install langchain-openai \
	&& pip install neo4j
 
WORKDIR app/
 
COPY ./app.py /app/app.py
 
CMD ["python", "app.py"]