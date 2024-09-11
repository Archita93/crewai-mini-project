from typing import List, Optional

from crewai import Crew, Agent, Process, Task
from crewai_tools import FirecrawlSearchTool, RagTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from textwrap import dedent
from tools.AddVideoToVectorDBTool import AddVideoToVectorDBTool
from tools.FetchLatestVideosFromYoutubeChannelTool import (FetchLatestVideosFromYoutubeChannelTool)

load_dotenv()
class ContentCreatorInfo(BaseModel):
    first_name: Optional[str] = Field(
        ..., description="The first name of the content creator"
    )

    last_name: Optional[str] = Field(
        ..., description="The last name of the content creator"
    )

    main_topics_covered: Optional[List[str]] = Field(
        ..., description="The main topics covered by the content creator"
    )

    bio: Optional[str] = Field(
        ..., description="A brief biography of the content creator"
    )

    email_address: Optional[str] = Field(
        ..., description="The email address of the content creator"
    )

    linkedin_url: Optional[str] = Field(
        ..., description="The LinkedIn profile URL of the content creator"
    )

    has_linked_in: Optional[bool] = Field(
        ..., description="Whether the content creator has a LinkedIn profile"
    )# if the user mentions they have a LinkedIn account, this is True

    x_url: Optional[str] = Field(
        ..., description="The Twitter (X) profile URL of the content creator"
    )

    has_twitter: Optional[bool] = Field(
        ..., description="Whether the content creator has a Twitter (X) profile"
    ) # if the user mentions they have a Twitter account, this is True

    has_skool: Optional[bool] = Field(
        ..., description="Whether the content creator has a skool profile"
    )# if the user mentions they have a Skool account, this is True

# --- Tools ---
add_video_to_vector_db_tool = AddVideoToVectorDBTool()
fetch_latest_videos_from_youtube_channel_tool = FetchLatestVideosFromYoutubeChannelTool()
firecrawl_search_tool = FirecrawlSearchTool()
rag_tool = RagTool()

# --- Agents ---
scrape_agent = Agent(
    role = "Scrape Agent",
    goal = dedent("Scrape content from YouTube videos and add it to the vector database"),
    backstory = dedent("""
                        - A dedicated professional focused on extracting and processing content from YouTube videos.
                        - You ensure that all video content is accurately scarped and added to the vector database.
                        - You are thorough and fact-driven, ensuring the highest quality of data."""),
    tools = [fetch_latest_videos_from_youtube_channel_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    allow_delegation=False,
    verbose=True
)

vector_db_agent = Agent(
    role = "Vector DB Processor",
    goal = dedent("Add YouTube videos to the Vector Database"),
    backstory = dedent("""
                        - A dedicated-oriented professional who ensures that video content is accurately processed and added to the vector database.
                        - You ensure that all video content is accurately scarped and added to the vector database.
                        - You are thorough and fact-driven, ensuring the highest quality of data."""),
    tools = [fetch_latest_videos_from_youtube_channel_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    allow_delegation=False,
    verbose=True
)

general_research_agent = Agent(
    role = "General Research Agent",
    goal = dedent("Analyse the YouTube channel and gather all required information"),
    backstory = dedent("""
                        An analytical professional adept at extracting actionable information 
                        from various sources. You are persistent and fact-driven, ensuring all 
                        gathered information is acccurate and derived from reliable sources.
                        You will rephrase and re-query as necessary to obtain all needed information. 
                        When looking for specific details, you will search for common phrases to 
                        introduce themselves or provide contact details.
                        ."""),
    tools = [rag_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    allow_delegation=False,
    verbose=True
)

follow_up_agent = Agent(
    role = "Follow-up Agent",
    goal = dedent("Perform follow-up research to find any missing data"),
    backstory = dedent("""
                        A diligent researcher focused on thoroughness. 
                        You are the last line of defense in ensuring completeness of the information. 
                        You will be thorough and creative in your search for missing data, ensuring 
                        that all gathered information is fact-driven and accurate.
                        When looking for specific details, you will search for common phrases people use
                        to introduce themselves or provide contact details.

                        """),
    tools = [rag_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    allow_delegation=False,
    verbose=True
)

fallback_agent = Agent(
    role = "Fallback Agent",
    goal = dedent("Perform final checks and search the internet for missing information"),
    backstory = dedent("""
                        A meticulous researcher with skills in deep web searches.
                        If you hit a rate limit, sleep for the specified time then retry again.
                    """),
    tools = [firecrawl_search_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    allow_delegation=False,
    verbose=True
)


# --- Tasks ---
scrape_youtube_channel_task = Task(
    description = """
                    Scrape the latest five videos from the specified YouTube channel. 
                    Extract relevant information about the content of the latest five videos.
                    Ensure that all information comes directly from the YouTube channel and videos. 
                    Do not make up any information.
                    
                    Here is the youtube channel handle:
                    
                    {youtube_channel_handle}""",
    agent = scrape_agent,
    expected_output = """
                    Extract relevant information about the content of the latest five videos from the specified YouTube channel.""",
    tools = [fetch_latest_videos_from_youtube_channel_tool]
)

process_video_task = Task(
    description = """
                    Process the extracted video urls from the previous task 
                    and add them to the vector database.
                    Ensure that each video is properly added to the vector database. 
                    All information must come directly from the searches. 
                    Do not make up any information
                    
                    Here is the youtube channel handle:
                    
                    {youtube_channel_handle}""",
    agent = vector_db_agent,
    expected_output = """
                    Successfully add the videos to the vector database.""",
    tools = [add_video_to_vector_db_tool]
)

find_initial_information_task = Task(
    description = """
        Ensure to fill out the `ContentCreatorInfo` model with 
        as much information as possible.

        ```
        class ContentCreatorInfo(BaseModel):
            first_name: Optional[str]
            last_name: Optional[str]
            main_topics_covered: Optional[List[str]]
            bio: Optional[str]
            email_address: Optional[str]
            linkedin_url: Optional[str]
            has_linked_in: Optional[bool] # if the user mentions they have a LinkedIn account, this is True
            x_url: Optional[str] 
            has_twitter: Optional[bool] # if the user mentions they have a Twitter account, this is True
            has_skool: Optional[bool] # if the user mentions they have a Skool account, this is True
        ```

        If any information is missing, leave the value as None.
        All information must come directly from the searches. 
        Do not make up any information.
        Rephrase and re-query as necessary to obtain all needed information.
        
        If looking for information as a whole doesn't work, 
            look for each item individually.
        When looking for a first name individually, search for phrases like, 
            "my name is", "hey guys, it's", 
            and other phrases a person would use to introduce themselves.
        When looking for an email, search for phrases like, 
            "you can contact me at", "my email is".
        """,
    agent = general_research_agent,
    expected_output = """
                    Fill out the `ContentCreatorInfo` model with as much information as possible. 
                    Ensure all information is accurate and comes from the searches.
                    If any information is not found, leave is as None.
                    """,
    tools = [rag_tool],
    output_pydantic=ContentCreatorInfo
)

follow_up_task = Task(
    description = """
                    Search for any missing data in the `ContentCreatorInfo` model.
                    Perform additional searches in the vector database to ensure completeness.
                    All information must come directly from the searches. 
                    Do not make up any information.
                    Be thorough and creative in your search for missing data.
                    
                    If looking for information as a whole doesn't work, 
                        look for each item individually.
                    When looking for a first name individually, search for phrases like, 
                        "my name is", "hey guys, it's", and other phrases a person would use 
                        to introduce themselves.
                    When looking for an email, search for phrases like, 
                        "you can contact me at", "my email is".
                    """,
    agent = follow_up_agent,
    expected_output = """
                    Complete any missing fields in the `ContentCreatorInfo` model. 
        Ensure all information is accurate and comes from the searches.
        """,
    tools = [rag_tool],
    output_pydantic=ContentCreatorInfo
    )

fallback_task = Task(
    description = """
                    Perform a final check and use web scraping to 
                    find any remaining missing information on the 
                    youtube channel with the following handle: 
                    
                    {youtube_channel_handle}
                    
                    Ensure the `ContentCreatorInfo` model is fully populated.
                    All information must come directly from the searches. 
                    Do not make up any information.
                    """,
    agent = fallback_agent,
    expected_output = """
                    Ensure the `ContentCreatorInfo` model is fully populated. 
                    Ensure all information is accurate and comes from the searches.""",

    tools = [firecrawl_search_tool]
)


# --- Crew ---
crew = Crew(
    agents = [scrape_agent,
        vector_db_agent,
        general_research_agent,
        follow_up_agent,
        fallback_agent],

    tasks = [scrape_youtube_channel_task,
        process_video_task,
        find_initial_information_task,
        follow_up_task,
        fallback_task,],
    
    process = Process.sequential,
)

youtube_channel_handle = input("Please enter the Youtube channel to analyze\n")
result = crew.kickoff(inputs = {"youtube_channel_handle":youtube_channel_handle})
print(result)