import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai

import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import os

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(
    page_title="AI Video Summarizer Tool",
    page_icon="🎥",
    layout="wide"
)

st.title("AI Video Summarizer Tool 🎥")
st.write("This tool can summarize uploaded vidoes as well as videos from YouTube. It is powered by Gemini.")


@st.cache_resource
def initialize_gemini_agent():
    return Agent(
        name="AI Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initialize the agent
video_summarizer_agent=initialize_gemini_agent()

video_type_selection = st.selectbox('Please Select Input Type', ['Enter YouTube URL', 'Upload a video'])

if video_type_selection == 'Upload a video':

    # File uploader
    uploaded_video_file = st.file_uploader(
        "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for summarization"
    )

    if uploaded_video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_summarization_video:
            temp_summarization_video.write(uploaded_video_file.read())
            summarization_video_path = temp_summarization_video.name

        st.video(summarization_video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the video."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        # Upload and process video file
                        processed_video = upload_file(summarization_video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        # Prompt generation for analysis
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )

                        # AI agent processing
                        response = video_summarizer_agent.run(analysis_prompt, videos=[processed_video])

                    # Display the result
                    st.subheader("Results of Video Analysis")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    # Clean up temporary video file
                    Path(summarization_video_path).unlink(missing_ok=True)
    else:
        st.info("Upload a video file to begin analysis.")

    # Customize text area height
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif video_type_selection == 'Enter YouTube URL':
    # Input for YouTube video URL
    video_url = st.text_input(
        "Enter a YouTube video URL",
        placeholder="Paste the URL of the video you'd like to summarize",
        help="The app will analyze the content of the video based on your query."
    )

    if video_url:
        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the video."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Analyzing video and gathering insights..."):
                        # Generate the analysis prompt
                        analysis_prompt = (
                            f"""
                            Analyze the YouTube video at the following URL: {video_url}.
                            Respond to the following query using the video content and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )

                        # AI agent processing
                        response = video_summarizer_agent.run(analysis_prompt)

                    # Display the result
                    st.subheader("Results of Video Analysis")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
    else:
        st.info("Enter a YouTube video URL to begin analysis.")

    # Customize text area height
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
