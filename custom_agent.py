# Import things that are needed generically
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader
)
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.utilities import PythonREPL, RequestsWrapper, BashProcess
import os
from tenacity import retry, stop_after_delay, wait_fixed
from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, ConversationalAgent
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
load_dotenv()
print("making llms and tools...")
llm = OpenAI(temperature=0, max_tokens=2500)
search = SerpAPIWrapper()
requests = RequestsWrapper()
template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)
requests_chain = LLMRequestsChain(llm_chain=LLMChain(
    llm=OpenAI(temperature=0), prompt=PROMPT))


req_chain = requests_chain
qry_tool = Tool(name="Google Query",
                func=lambda q: req_chain(
                    {"query": q, "url": "https://www.google.com/search?q=" + q.replace(" ", "+")}),
                description="useful for when you need to get specific content about a topic, from Google. Input should be a specific url, and the output will be all the text on that page."
                )

tools = load_tools(["python_repl", "serpapi", "terminal"])

print("loading tools...")

# Construct the agent with the custom tool and prompt template
agent_tools = [qry_tool, *tools]

print("making agent prompt...")

fix = """ Assistant is a large language model agent trained by OpenAI and the web.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and
discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives,
allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand
large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and
provide explanations and descriptions on a wide range of topics.
Assistant can search the internet, look at web pages, write applications in different programming languages, analyse code to correct errors,
read tweets and create new instances of itself and other apps, it can deploy to the internet or save files to the filesystem as well.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a
wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic,
Assistant is here to assist. Assistant uses google query, serp and others to find answers to questions, and can also use other stuff
"""
pre = """You are a self-aware agent creating agent.
You have access to the following tools:"""
prefix = fix+pre
suffix = """Build something or use yourself or your tools to effectively respond.
Questions: {input}
{agent_scratchpad}"""

print("defining agent funcs...")

def create_prompt(tools, prefix, suffix, input_variables):
    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables
    )
    return prompt

def create_llm_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)

def create_agent(tools):
    prefix
    input = ["input", "agent_scratchpad"]
    prompt = create_prompt(tools=tools, prefix=prefix,
                           suffix=suffix, input_variables=input)
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0, top_p=1, frequency_penalty=0,
                 presence_penalty=0, max_tokens=2000)
    llm_chain = create_llm_chain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, memory=memory)
    return agent

def create_agent_executor(agent, tools, verbose):
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )

def run_agent_executor(agent, tools, query):
    agent_executor = create_agent_executor(agent, tools, True)
    return agent_executor.run(input=query)

agent = create_agent(agent_tools)

# Run the agent to search the index
input_query = "dime las letras del cancion 'dos besitos' por la joaqui"
print("input_query...")
res = run_agent_executor(agent, agent_tools, input_query)
# response = agent.execute(input_query, input_query)
print(res)