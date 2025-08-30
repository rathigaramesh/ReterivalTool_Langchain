from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline_chain import run_chain_mode
from config import GOOGLE_API_KEY
import re

def build_agent(retrievers_map):
    tools = []
    for name, retriever in retrievers_map.items():
        # Use a closure to capture the retriever correctly
        def create_tool_func(retriever_obj):
            def tool_func(query):
                print(f"Tool {name} called with: '{query}'")
                
                # Clean the query from agent formatting
                if isinstance(query, dict):
                    clean_query = query.get('input', '') or query.get('query', '') or str(query)
                else:
                    clean_query = str(query)
                
                # Remove any quotes or special formatting
                clean_query = re.sub(r'^[\"\']|[\"\']$', '', clean_query)
                clean_query = clean_query.strip()
                
                print(f"Cleaned query: '{clean_query}'")
                
                if not clean_query or clean_query == "{}":
                    return "Please provide a valid question."
                
                try:
                    answer, sources = run_chain_mode(clean_query, retriever_obj)
                    return answer
                except Exception as e:
                    return f"Error in retriever: {str(e)}"
            return tool_func
        
        tool_func = create_tool_func(retriever)
        
        tools.append(
            Tool(
                name=f"{name}Retriever",
                func=tool_func,
                description=f"Useful for answering questions using the {name} retriever. Input should be a clear question."
            )
        )
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    return agent

def run_agent_mode(query, agent):
    try:
        print(f"Agent mode called with: '{query}'")
        result = agent.run({"input": query})
        return result
    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        print(error_msg)
        return error_msg