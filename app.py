# https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.prompts.prompt import PromptTemplate
# web server stuff
import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer


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


# Good query example:
# MATCH (m:Movie {name:"Top Gun"})<-[:ACTED_IN]-() RETURN count(*) AS numberOfActors
#chain.run("How many people played in Top Gun?")

# Good query example:
# MATCH (m:Movie {name:"Top Gun"})<-[:ACTED_IN]-(a:Actor) RETURN a.name AS actorName
#chain.run("Who played in Top Gun?")

# Good query example:
# MATCH (a:Actor {name:"Tom Cruise"})-[:ACTED_IN]->(m:Movie) RETURN m.name as movieName
#chain.run("What movies did Tom Cruise play in?")




def query_graph(user_input):
    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        validate_cypher=True,
        return_intermediate_steps=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )
    result = chain(user_input)
    return result


st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])    

with title_col:
    st.title("Cypher Database Querier")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

with st.expander("Open to see examples"):
    c = st.container()
    c.write("How many people played in Top Gun?")
    c.write("Who played in Top Gun?")
    c.write("What movies did Tom Cruise play in?")

user_input = st.text_input("Enter your question", key="input")
if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            result = query_graph(user_input)
           
            # Note: if we don't have 'return_intermediate_steps=True' set in GraphCypherQAChain then the below lines will blow us up...
            intermediate_steps = result["intermediate_steps"]
            cypher_query = intermediate_steps[0]["query"]
            database_results = intermediate_steps[1]["context"]

            answer = result["result"]
            st.session_state.system_msgs.append(answer)
        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col2:
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
        
    with col3:
        if database_results:
            st.text_area("Last Database Results", database_results, key="_database", height=240)








## Copied from https://huggingface.co/spaces/santiferre/procesamiento-lenguaje-natural/blob/main/app.py
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
