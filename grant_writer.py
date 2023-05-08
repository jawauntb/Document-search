# Here's a new code snippet that combines the Document Searcher and Custom Agent to generate grant proposals based on user input:

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from doc_searcher import get_answer_from_vector, llm, vector_store, chat_history
from custom_agent import agent as custom_agent, agent_tools

load_dotenv()

# Modify this variable with a relevant topic to generate a grant proposal
input_topic = "climate change mitigation"

# Ask the Document Searcher for information about the input topic
doc_search_question = f"What are some successful grant proposals for {input_topic}?"
doc_search_answer = get_answer_from_vector(doc_search_question, llm, vector_store, chat_history)

# Use the Custom Agent to generate a grant proposal based on the information retrieved
agent_input = f"Based on the following information about successful grant proposals for {input_topic}, create a new grant proposal:\n{doc_search_answer}"
agent_executor = AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=agent_tools, verbose=True)
generated_grant_proposal = agent_executor.run(input=agent_input)

print("\nGenerated Grant Proposal:\n")
print(generated_grant_proposal)

# This script first asks the Document Searcher for information about successful grant proposals related to the specified input_topic. It then uses the Custom Agent to generate a new grant proposal based on the information retrieved from the Document Searcher. The final generated grant proposal is printed to the console.

# Remember to replace input_topic with the topic you want to generate a grant proposal for. You can also modify the doc_search_question and agent_input variables to ask more specific questions or provide additional context to the Custom Agent.