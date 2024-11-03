import os
import sys

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

default_llm = AzureChatOpenAI(
    api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt35"),
    azure_endpoint=os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://<your-endpoint>.openai.azure.com/"
    ),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
)

llm = LLM(
    model="azure/gpt-4o-mini",
    temperature=0,
    max_tokens=2048,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42,
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

# Create a researcher agent
researcher = Agent(
    role="Senior Researcher",
    goal="Discover groundbreaking technologies",
    verbose=True,
    llm=llm,
    backstory="A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.",
)

# Task for the researcher
research_task = Task(
    description="Identify the next big trend in AI",
    expected_output="A detailed report on the next big trend in AI",
    agent=researcher,  # Assigning the task to the researcher
)


# Instantiate your crew
tech_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,  # Tasks will be executed one after the other
    verbose=True,
    name="Tech Crew",
    llm=llm,
)

# Begin the task execution
tech_crew.kickoff()
