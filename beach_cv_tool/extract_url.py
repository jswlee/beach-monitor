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

def get_youtube_livestream_url(search_query: str) -> tuple[str | None, str | None]:
    """
    Searches YouTube using the official API and returns the URL and title of the first result.

    Args:
        search_query: The title or key search terms for the video.

    Returns:
        Tuple of (video_url, video_title), or (None, None) if no results are found.
    """
    if not YOUTUBE_API_KEY:
        print("❌ ERROR: YouTube API key not found.")
        return None, None

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
            return None, None

        # Get the video ID and construct the full URL
        first_result = search_results[0]
        video_id = first_result["id"]["videoId"]
        video_title = first_result["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        print(f"✅ Found video: {video_title}")
        return video_url, video_title

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def validate_video_title(video_title: str, search_query: str) -> bool:
    """
    Check if all words from search query are present in video title.
    
    Args:
        video_title: The title of the video found
        search_query: The original search query
        
    Returns:
        True if all words from search query are in the title (case-insensitive)
    """
    if not video_title or not search_query:
        return False
    
    # Split search query into words and convert to lowercase
    search_words = search_query.lower().split()
    title_lower = video_title.lower()
    
    # Check if all search words are in the title
    return all(word in title_lower for word in search_words)


if __name__ == "__main__":
    search_query = "LIVE 24/7 4K MAUI LIVE CAM WhalerCondo.net"
    url, title = get_youtube_livestream_url(search_query)
    
    if url:
        print(f"\nYouTube Stream URL: {url}")
        print(f"Video Title: {title}")
        print(f"Title Valid: {validate_video_title(title, search_query)}")