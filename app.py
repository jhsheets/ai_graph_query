# https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.prompts.prompt import PromptTemplate

# Tell Ollama which model we want to use.
# Ollama has to have this model downloaded and available
# TODO: pass this in as an env var
#
# Note: I've tried using starcoder2, but it didn't produce good results on its own and was slow
model_id = 'llama2'

# Our LLM to use. Here we have a link to our Ollama REST API
llm = Ollama(base_url='http://host.docker.internal:11434', model=model_id)

# Our Neo4j graph db
# TODO: pass these in as an env var
graph = Neo4jGraph(
    url="bolt://host.docker.internal:7687", username="neo4j", password="pleaseletmein"
)

# Seed the database 
graph.query(
    """
    MERGE (m:Movie {name:"Top Gun"})
    WITH m
    UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
    MERGE (a:Actor {name:actor})
    MERGE (a)-[:ACTED_IN]->(m)
    """
    )

# I think this is necessary for our {schema} variable below to resolve
graph.refresh_schema()

print(graph.schema)


# Note that there are two input variables here: {schema} and {question}
# The {question} value will be supplied when we execute chain.run(...), but what about {schema}? How is it set?
# Looking at the source code for GraphCypherQAChain (https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/graph_qa/cypher.py#L215)
# it looks like it will use the langchain_conmmunity.graphs.Neo4jGraph's get_structured_schema field to get the schema.
# When chain.run(...) is executed, it will set the {schema} variable to its value.
# 
# There's no way to alter how GraphCypherQAChain gets the schema. If we didn't want to use it's way of fetching the schema,
# we could probably change the prompt template below to not include a {schema} variable. We could query the database however we liked
# to get the schema and just populate the template before passing it in (or see if we can pass our own parameters in somehow)
# 
# That said, it looks like GraphCypherQAChain always tries to query the database for the schema. If we tried using this library 
# on another graph db (like Posgtres AGE) it would probably blow up.
CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many people played in Top Gun?
MATCH (m:Movie {{name:"Top Gun"}})<-[:ACTED_IN]-()
RETURN count(*) AS numberOfActors

The question is:
{question}
"""


# Pass in our template as a prompt; define our variables
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Construct our GraphCypherQAChain object
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    validate_cypher=False,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

# Good query example:
# MATCH (m:Movie {name:"Top Gun"})<-[:ACTED_IN]-() RETURN count(*) AS numberOfActors
chain.run("How many people played in Top Gun?")

# Good query example:
# MATCH (m:Movie {name:"Top Gun"})<-[:ACTED_IN]-(a:Actor) RETURN a.name AS actorName
chain.run("Who played in Top Gun?")

# Good query example:
# MATCH (a:Actor {name:"Tom Cruise"})-[:ACTED_IN]->(m:Movie) RETURN m.name as movieName
chain.run("What movies did Tom Cruise play in?")


## Copied from https://huggingface.co/spaces/santiferre/procesamiento-lenguaje-natural/blob/main/app.py
#import streamlit as st
#from streamlit_chat import message
#from timeit import default_timer as timer
#
#from langchain.graphs import Neo4jGraph
#from langchain.chains import GraphCypherQAChain
#from langchain.prompts.prompt import PromptTemplate
#from langchain.chat_models import AzureChatOpenAI
#
#import dotenv
#import os
#
#dotenv.load_dotenv()
#
## OpenAI API configuration
#llm = AzureChatOpenAI(
#    deployment_name = "chat-gpt4",
#    openai_api_base = os.getenv("OPENAI_API_BASE"),
#    openai_api_version = os.getenv("OPENAI_API_VERSION"),
#    openai_api_key = os.getenv("OPENAI_API_KEY"),
#    openai_api_type = "azure",
#    temperature = 0
#)
#
##Neo4j configuration
#neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
#neo4j_user = os.getenv("NEO4J_USER")
#neo4j_password = os.getenv("NEO4J_PASSWORD")
#
## Cypher generation prompt
#cypher_generation_template = """
#You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
#1. Generate Cypher query compatible ONLY for Neo4j Version 5
#2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
#3. Use only Nodes and relationships mentioned in the schema
#4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Client, use `toLower(client.id) contains 'neo4j'`. To search for Slack Messages, use 'toLower(SlackMessage.text) contains 'neo4j'`. To search for a project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.)
#5. Never use relationships that are not mentioned in the given schema
#6. When asked about projects, Match the properties using case-insensitive matching and the OR-operator, E.g, to find a logistics platform -project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.
#
#schema: {schema}
#
#Examples:
#Question: Which client's projects use most of our people?
#Answer: ```MATCH (c:CLIENT)<-[:HAS_CLIENT]-(p:Project)-[:HAS_PEOPLE]->(person:Person)
#RETURN c.name AS Client, COUNT(DISTINCT person) AS NumberOfPeople
#ORDER BY NumberOfPeople DESC```
#Question: Which person uses the largest number of different technologies?
#Answer: ```MATCH (person:Person)-[:USES_TECH]->(tech:Technology)
#RETURN person.name AS PersonName, COUNT(DISTINCT tech) AS NumberOfTechnologies
#ORDER BY NumberOfTechnologies DESC```
#
#Question: {question}
#"""
#
#cypher_prompt = PromptTemplate(
#    template = cypher_generation_template,
#    input_variables = ["schema", "question"]
#)
#
#CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
#The information part contains the provided information that you must use to construct an answer.
#The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
#Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
#If the provided information is empty, say that you don't know the answer.
#Final answer should be easily readable and structured.
#Information:
#{context}
#
#Question: {question}
#Helpful Answer:"""
#
#qa_prompt = PromptTemplate(
#    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
#)
#
#def query_graph(user_input):
#    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
#    chain = GraphCypherQAChain.from_llm(
#        llm=llm,
#        graph=graph,
#        verbose=True,
#        return_intermediate_steps=True,
#        cypher_prompt=cypher_prompt,
#        qa_prompt=qa_prompt
#        )
#    result = chain(user_input)
#    return result
#
#
#st.set_page_config(layout="wide")
#
#if "user_msgs" not in st.session_state:
#    st.session_state.user_msgs = []
#if "system_msgs" not in st.session_state:
#    st.session_state.system_msgs = []
#
#title_col, empty_col, img_col = st.columns([2, 1, 2])    
#
#with title_col:
#    st.title("Conversational Neo4J Assistant")
#with img_col:
#    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)
#
#user_input = st.text_input("Enter your question", key="input")
#if user_input:
#    with st.spinner("Processing your question..."):
#        st.session_state.user_msgs.append(user_input)
#        start = timer()
#
#        try:
#            result = query_graph(user_input)
#            
#            intermediate_steps = result["intermediate_steps"]
#            cypher_query = intermediate_steps[0]["query"]
#            database_results = intermediate_steps[1]["context"]
#
#            answer = result["result"]
#            st.session_state.system_msgs.append(answer)
#        except Exception as e:
#            st.write("Failed to process question. Please try again.")
#            print(e)
#
#    st.write(f"Time taken: {timer() - start:.2f}s")
#
#    col1, col2, col3 = st.columns([1, 1, 1])
#
#    # Display the chat history
#    with col1:
#        if st.session_state["system_msgs"]:
#            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
#                message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
#                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")
#
#    with col2:
#        if cypher_query:
#            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
#        
#    with col3:
#        if database_results:
#            st.text_area("Last Database Results", database_results, key="_database", height=240)
