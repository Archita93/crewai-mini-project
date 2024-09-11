from typing import Type

from crewai_tools.tools.base_tool import BaseTool
from dotenv import load_dotenv
from embedchain import App
from embedchain.models.data_type import DataType
from pydantic.v1 import BaseModel, Field

class AddVideoToVectorDBInput(BaseModel):

    video_url: str = Field(
        ..., decription="the URL of the YouTube video to add to the vector DB."
    )

class AddVideoToVectorDBOutput(BaseModel):
    videos: bool = Field(
        ..., decription="Whether the video was successfully added to the vector DB."
    )

class AddVideoToVectorDBTool(BaseTool):
    name: str = "Add Video to Vector DB"
    description: str = (
        "Adds a YouTube video to the Vector Database."
    )
    args_schema: Type[BaseModel] = AddVideoToVectorDBInput
    return_schema: Type[BaseModel] = AddVideoToVectorDBOutput

    def _run(
        self,
        video_url: str,
    ) -> AddVideoToVectorDBOutput:

        try:
            app = App()   # EmbedChain
            app.add(video_url, data_type=Datatype.YOUTUBE_VIDEO)
            return AddVideoToVectorDBOutput(sucess=True)
        except Exception as e:
            return AddVideoToVectorDBOutput(sucess=False)
    
        
