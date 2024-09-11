from typing import List, Optional

from crewai import Crew, Agent, Process, Task
from crewai_tools import FirecrawlSearchTool, RagTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from textwrap import dedent
from tools.AddVideoToVectorDBTool import AddVideoToVectorDBTool
from tools.FetchLatestVideosFromYoutubeChannelTool import (FetchLatestVideosFromYoutubeChannelTool)
