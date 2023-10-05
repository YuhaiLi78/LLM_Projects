from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool

from constants import OPENAI_API_KEY

if __name__ == '__main__':
    python_agent_executor = create_python_agent(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, 
                                                               temperature=0, 
                                                               model='gpt-4'),
                                                tool=PythonREPLTool(),
                                                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                                verbose=True)
    
    query = 'create and save to current directory 3 QR codes that point to website https://www.yuhaili.com/'
    python_agent_executor.run(query)