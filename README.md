# Beach Monitor

**Goal:** A conversational AI agent that provides real-time analysis of beach conditions at Kāʻanapali Beach, Maui, by analyzing a live YouTube stream.

This project uses a fine-tuned YOLOv8 model to detect people and boats, providing users with a summary of beach activity through a simple and intuitive chat interface.

---

## Features

- **Conversational AI Agent**: Built with LangGraph, the agent can understand natural language queries and provide informative responses.
- **Real-time Analysis**: Captures snapshots from a live YouTube stream to provide up-to-the-minute beach conditions.
- **Object Detection**: Uses a fine-tuned YOLOv8 model to accurately detect and count people and boats.
- **Interactive UI**: A simple and clean chat interface built with Streamlit, allowing users to ask questions and receive real-time analysis.
- **Annotated Images**: The agent can provide an annotated image with bounding boxes showing the detected objects, offering a visual confirmation of the analysis.

---

## Architecture

The project is designed with a modular architecture that separates the different components of the system:

```
User (Streamlit UI) -> LangGraph Agent -> Beach Monitoring Tool -> Snapshot Capture -> YOLO Detection
```

- **Streamlit UI (`ui/chat.py`)**: The user-facing chat interface that communicates with the agent.
- **LangGraph Agent (`agent/graph.py`)**: The conversational AI that processes user queries and orchestrates the tool calls.
- **Beach Monitoring Tool (`agent/tools.py`)**: A tool that integrates the computer vision components, providing a simple interface for the agent to get beach status.
- **Snapshot Capture (`beach_cv_tool/capture.py`)**: Captures snapshots from the live YouTube stream.
- **YOLO Detection (`beach_cv_tool/detection.py`)**: Runs the fine-tuned YOLOv8 model on the captured snapshots to detect and count people and boats.

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key for the LangGraph agent.
- A YouTube API key for fetching the livestream URL.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd beach-monitor
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your API keys:**

    Create a `.env` file in the root of the project and add your API keys:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    YOUTUBE_API_KEY="your-youtube-api-key"
    ```

### Running the Application

To start the application, run the following command from the root of the project:

```bash
python main.py
```

This will launch the Streamlit UI, and you can start interacting with the agent in your browser at `http://localhost:8501`.

---

## Usage

Once the application is running, you can ask the agent questions about the beach conditions, such as:

-   "How busy is the beach right now?"
-   "Are there many people at the beach?"
-   "Can you show me the current conditions?"

After providing a summary, the agent will ask if you want to see the annotated image. If you respond with "yes" or a similar affirmative, the image with bounding boxes will be displayed in the chat.

---

## Configuration

The project's behavior can be configured through the `config.yaml` file. This includes settings for:

-   **Camera**: The YouTube stream URL, capture interval, and retries.
-   **Models**: Paths to the YOLO model and detection settings.
-   **Agent**: The language model to use, temperature, and system prompt.
-   **Data Paths**: Directories for snapshots and other data.

---

## File Descriptions

-   **`main.py`**: The main entry point for the application, responsible for launching the Streamlit UI.
-   **`config.yaml`**: The main configuration file for the project.
-   **`requirements.txt`**: A list of all the Python dependencies required for the project.
-   **`.env`**: A file for storing your API keys and other environment variables.

### Agent (`agent/`)

-   **`graph.py`**: Defines the LangGraph agent, including its state, nodes, and edges.
-   **`tools.py`**: Contains the `BeachMonitoringTool`, which integrates the computer vision components and provides a simple interface for the agent.

### Computer Vision (`beach_cv_tool/`)

-   **`capture.py`**: Handles the capturing of snapshots from the YouTube livestream.
-   **`detection.py`**: Runs the YOLOv8 model to detect people and boats, and saves an annotated image.
-   **`extract_url.py`**: Fetches the direct URL of the YouTube livestream using the YouTube API.
-   **`livestream_capture_youtube.py`**: A utility script for capturing video from YouTube streams.

### UI (`ui/`)

-   **`chat.py`**: The Streamlit application that provides the user-facing chat interface.
