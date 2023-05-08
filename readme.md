
Document Searcher and Custom Agent
This is a guide to help you set up and run the Document Searcher and Custom Agent using Visual Studio Code (VSCode). The Document Searcher searches through a folder of text documents to answer questions, and the Custom Agent can be used to search the internet or interact with other tools.

Prerequisites
Python 3.7 or higher
Visual Studio Code
Installation
Clone the repository from the provided link.
bash
Copy code
git clone https://replit.com/@JawaunBrown/Document-searcher
Open the cloned repository in Visual Studio Code.

Create a virtual environment:

Copy code
python -m venv venv
Activate the virtual environment:
On Windows:
Copy code
.\venv\Scripts\activate
On macOS/Linux:
bash
Copy code
source venv/bin/activate
Install the required packages:
Copy code
pip install langchain openai chroma dotenv tenacity llama_index serpapi
Setup
Obtain an OpenAI API key by signing up for an account at https://beta.openai.com/signup/.

Create a .env file in the project root directory and add the OpenAI API key:

makefile
Copy code
OPENAI_API_KEY=your_api_key_here
Change the doc_folder variable in doc_searcher.py to point to the directory containing your grant documents:
python
Copy code
doc_folder = '/path/to/your/grant/documents'
Usage
Document Searcher
Run the doc_searcher.py script to search the documents for answers to your questions:

Copy code
python doc_searcher.py
Modify the questions list in doc_searcher.py to ask different questions:

python
Copy code
questions = ["what is", "how to", "why is"]
Custom Agent
Run the custom_agent.py script to interact with the Custom Agent:

Copy code
python custom_agent.py
Modify the input_query variable in custom_agent.py to ask different questions or use different tools:

python
Copy code
input_query = "what is the capital of France?"
Advanced
If you want to use the Custom Agent with the Document Searcher or integrate other custom tools, you can modify the custom_agent.py script accordingly. The provided code snippets should give you a good starting point.