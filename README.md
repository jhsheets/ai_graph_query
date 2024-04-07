# Overview
This is just a learning example using a local AI to generate Cypher queries against a graph database.

# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open ollama web ui: http://localhost:3002
4. Click create an account and set values
5. Click on `settings > models > pull a model from ollama.com`
6. Enter `llama2` and press download
7. Go to Docker's console output for `langchain-1` and verify it ran

You can also access the Neo4J web UI at: http://localhost:7474

# Behavior
I don't always get correct results.
Sometimes it will generate correct queries, and sometimes it doesn't.
Sometimes it will generate a good query, then a bad one when re-runinng the test.
I've tried adding some extra examples to the prompt template, but it's causing it to blow up running any query at all.
This probably requires better training, or fine tuning to get better results, or maybe some reinforced learning.

# TODO
1. Use Streamlit in my Dockerfile to serve up a webserver I can query instead of having hard-coded questions
2. Try using this LLM model instead: https://huggingface.co/monsterapi/llama2-code-generation
3. Use a different way to download the LLM file instead of requiring using ollama-ui
4. Fix docker compose so the langchain app will properly wait on graphdb and llama to be available, or at least doesn't crash if theyr'e not up (yet)
5. Use a more complex database example
6. Try switching from Neo4J to Postgres with Apache AGE
