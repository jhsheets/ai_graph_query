# Overview
This is just a learning example using a local AI to generate Cypher queries against a graph database.

# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open ollama web ui: http://localhost:3002
4. Click create an account and set values
5. Click on `settings > models > pull a model from ollama.com`
6. Enter `llama2` and press download
7. Open web page: http://localhost:8080
8. Enter a question. There's an expandable section to show example queries

You can also access the Neo4J web UI at: http://localhost:7474 if you want to run queries directly

# Behavior
I don't always get correct results.
Sometimes it will generate correct queries, and sometimes it doesn't.
Sometimes it will generate a good query, then a bad one when re-runinng the test.
I've tried adding some extra examples to the prompt template, but it's causing it to blow up running any query at all.
This probably requires better training, or fine tuning to get better results, or maybe some reinforced learning.

# TODO
1. Try using this LLM model instead: https://huggingface.co/monsterapi/llama2-code-generation
   1. I think we can replace the langchain LLMChain with a huggingface pipeline: https://stackoverflow.com/questions/77152888/huggingfacepipeline-and-langchain
3. Use a different way to download the LLM file instead of requiring using ollama-ui
4. Use a more complex database example
5. Try switching from Neo4J to Postgres with Apache AGE
