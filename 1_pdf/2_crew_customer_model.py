from crewai import Crew, Agent, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

# --- Tools ---

# control over the embedder, response and retrieve data
pdf_search_tool = PDFSearchTool(
    pdf="example_home_inspection.pdf",
    config = {
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": "claude-3-haiku"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1"
        }
    }
}

)

# -- Agents ---
research_agent = Agent(
    role = "Research Agent",
    goal = dedent("Search through the PDF to find relevant answers"),
    backstory = dedent("The research agent is adept at searching and extracting data from documents, ensuring accurate and prompt responses."),
    tools = [pdf_search_tool],
    allow_delegation=False,
    verbose=True
)

professional_writer_agent = Agent(
    role = "Professional Writer Agent",
    goal = dedent("Write professional emails based on the research agent's findings"),
    backstory = dedent("The professional writer agent has excellent writing skills and is able to craft clear and concise emails based on the provided information."),
    tools = [],
    allow_delegation=False,
    verbose=True

)

# --- Tasks ---
answer_customer_question_task = Task(
    description="""Answer the customer's question based on the home inspection PDF.
                    The research agent will search through the PDF to find the relevant answers. 
                    Your final answer must be clear and accurate, based on the content of the home inspection PDF.

                    Here is the customer's question:
                    {customer_question}
                    """,
    agent=research_agent,
    tools = [pdf_search_tool],
    expected_output="""Provide clear and concise answers to the customer's questions based on the content of the home inspection PDF.""",
)

write_email_task = Task(
    description="""
                    - Write a professional email to a contractor based on the research agent's findings.
                    - The email should clearly state the issues found in the spcified section of the report
                    and request a quote or action plan for fixing these issues.
                    - Ensure the email is signed with the following details:

                        Best regards, 

                        Archita Srivastava, 
                        Hancock Realty
                    {customer_question}
                    """,
    agent=professional_writer_agent,
    tools = [],
    expected_output="""Write a clear and concise email that can be sent to a contractor to address the issues found in the home inspection report.""",
)

# --- Crew ---
crew = Crew(
    agents = [research_agent,professional_writer_agent],
    tasks = [answer_customer_question_task,write_email_task],
    process = Process.sequential
)

customer_question = input("Which section of the report would you like to generate a work order for?\n")

result = crew.kickoff_for_each(inputs=[{"customer_question":"Roof"}, {"customer_question":"Electrical"}])
print(result)