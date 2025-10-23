import os
import streamlit as st
import requests
from io import StringIO

from dotenv import load_dotenv, find_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool

load_dotenv(find_dotenv())
# --- 1. CONFIGURATION ---
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")

DATA_ENDPOINTS = {
    "crop_production": f"https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de?api-key={DATA_GOV_API_KEY}",
    "rainfall": f"https://api.data.gov.in/resource/6c05cd1b-ed59-40c2-bc31-e314f39c6971?api-key={DATA_GOV_API_KEY}"
}

# --- 2. TOOL DEFINITIONS ---

@tool
def fetch_live_data(dataset_name: str) -> str:
    """
    Fetches live data from the data.gov.in API.
    Input must be one of: 'crop_production' or 'rainfall'.
    Returns the raw data as a text string in CSV format.
    """
    if dataset_name not in DATA_ENDPOINTS:
        return f"Error: Unknown dataset '{dataset_name}'. Available: {list(DATA_ENDPOINTS.keys())}"
        
    url = DATA_ENDPOINTS[dataset_name]
    
    params = {
        "format": "csv",
        "limit": 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data_string = response.text
        source_citation = f"Source: {dataset_name} dataset from data.gov.in"
        
        return f"{source_citation}\n\n{data_string}"
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {str(e)}"

# Python analysis tool
python_repl_tool = PythonREPLTool()

# --- 3. AGENT SETUP ---

def setup_agent():
    """
    Initializes the LangChain ReAct agent with proper syntax for modern LangChain.
    """
    # Initialize LLM
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set!")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Best balance of speed and capability
        temperature=0,
        google_api_key=google_api_key,
        request_timeout=60,
        max_retries=2
    )
    
    # Define tools
    tools = [fetch_live_data, python_repl_tool]
    
    # Create ReAct prompt template
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT INSTRUCTIONS:
1. To fetch data, use the fetch_live_data tool with dataset names: 'crop_production' or 'rainfall'
2. To analyze data, use the Python_REPL tool with pandas code
3. When using Python_REPL to read CSV data from fetch_live_data output:
   - Import: from io import StringIO; import pandas as pd
   - Skip the first 2 lines (citation): pd.read_csv(StringIO(data_string), skiprows=2)
4. Always cite your sources in the Final Answer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Project Samarth (Live Data)", layout="wide")
st.title("🇮🇳 Project Samarth: Intelligent Q&A (Live Data)")
st.markdown("Ask complex questions about India's agricultural and climate data.")

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY environment variable not set. Please set it before running.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Samarth Agent..."):
        try:
            st.session_state.agent_executor = setup_agent()
            st.success("✅ Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar with sample questions
st.sidebar.header("📝 Sample Questions")
sample_questions = [
    "What are the top 3 crops produced in Maharashtra?",
    "Compare rainfall patterns between Maharashtra and Gujarat",
    "Which district has the highest sugarcane production?"
]

for q in sample_questions:
    if st.sidebar.button(q, key=f"sample_{hash(q)}"):
        st.session_state.last_question = q

# Handle input
if "last_question" in st.session_state and st.session_state.last_question:
    user_input = st.session_state.last_question
    st.session_state.last_question = None
else:
    user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Samarth is analyzing..."):
            try:
                response = st.session_state.agent_executor.invoke({"input": user_input})
                answer = response.get('output', 'No response generated')
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors specifically
                if "429" in error_str or "quota" in error_str.lower():
                    error_message = """❌ **Rate Limit Exceeded**
                    
You've hit the Gemini API rate limit. Here's what to do:

1. **Wait 45-60 seconds** before trying again
2. **Switch to a different model** (gemini-1.5-flash has higher limits)
3. **Get a new API key** from https://aistudio.google.com/apikey
4. **Upgrade to paid tier** for higher limits

The free tier limits are:
- 15 requests per minute
- 1 million tokens per day
- 1500 requests per day"""
                    st.error(error_message)
                else:
                    error_message = f"❌ Error: {error_str}\n\nPlease try rephrasing your question."
                    st.error(error_message)
                
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Be specific in your questions for better results!")