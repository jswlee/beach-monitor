import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from pathlib import Path

# Find the project root and load the .env file from there
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def get_youtube_livestream_url(search_query: str) -> str | None:
    """
    Searches YouTube using the official API and returns the URL of the first result.

    Args:
        search_query: The title or key search terms for the video.

    Returns:
        The URL of the first video result, or None if no results are found.
    """
    if not YOUTUBE_API_KEY:
        print("❌ ERROR: YouTube API key not found.")
        return None

    try:
        # Build the YouTube service object
        youtube_service = build(
            YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
        )

        # Call the search.list method to retrieve results
        search_response = youtube_service.search().list(
            q=search_query,
            part="snippet", # We only need basic info
            maxResults=1,   # Get only the top result
            type="video"    # Search only for videos
        ).execute()

        # Extract the results from the response
        search_results = search_response.get("items", [])

        if not search_results:
            print(f"❌ No video results found for '{search_query}'.")
            return None

        # Get the video ID and construct the full URL
        first_result = search_results[0]
        video_id = first_result["id"]["videoId"]
        video_title = first_result["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        print(f"✅ Found video: {video_title}")
        return video_url

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    search_query = "4K MAUI LIVE CAM WhalerCondo.net"
    url = get_youtube_livestream_url(search_query)
    
    if url:
        print(f"\nYouTube Stream URL: {url}")