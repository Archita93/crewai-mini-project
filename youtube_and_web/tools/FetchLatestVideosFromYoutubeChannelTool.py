import os
from datetime import datetime, timezone
from typing import List, Type

import requests
from crewai_tools.tools.base_tool import BaseTool
from pydantic.v1 import BaseModel, Field


class FetchLatestVideosFromYoutubeChannelInput(BaseModel):

    youtube_channel_handle: str = Field(
        ..., decription="the YouTube channel handle (e.g., '@channelhandle')."
    )
    max_results: int = Field(10, description = "The maximum number of results to return.")

class VideoInfo(BaseModel):
    video_id : str
    title : str
    publish_data : datetime
    video_url : str

class FetchLatestVideosFromYoutubeChannelOutput(BaseModel):
    videos: List[VideoInfo]

class FetchLatestVideosFromYoutubeChannelTool(BaseTool):
    name: str = "Fetch Latest Videos for Channel"
    description: str = (
        "Fetches the latest videos for a specified YouTube channel handle."
    )
    args_schema: Type[BaseModel] = FetchLatestVideosFromYoutubeChannelInput
    return_schema: Type[BaseModel] = FetchLatestVideosFromYoutubeChannelOutput

    def _run(
        self,
        youtube_channel_handle: str,
        max_results: int = 10,
    ) -> FetchLatestVideosFromYoutubeChannelOutput:
        api_key = os.getenv("YOUTUBE_API_KEY")

        # creating a request
        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part":"snippet",
            "type":"channel",
            "q":youtube_channel_handle,
            "key":api_key, 
        }
        response = requests.get(url,params=params)
        response.raise_for_status()
        items = response.json().get("items",[])
        if not items:
            raise ValueError(f"No channel found for handle {youtube_channel_handle}")
        
        channel_id = items[0]["id"]["channelId"]
        params = {
            "part":"snippet",
            "channelId": channel_id,
            "maxResults": max_results, 
            "order":"date",
            "type":"video",
            "key":api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        items = response.json().get("items",[])

        videos = []
        for item in items:
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            publish_date = datetime.fromisoformat(
                item["snippet"]["publishedAt"].replace("Z","+00:00")
            ).astimezone(timezone.utc)
            videos.append(
                VideoInfo(
                    video_id=video_id,
                    title=title,
                    publish_date=publish_date,
                    video_url=f"https://www.youtube.com/watch?v={video_id}"
                )
            )
        
        return FetchLatestVideosFromYoutubeChannelOutput(videos=videos)

        
