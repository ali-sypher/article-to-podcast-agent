import os
import uuid
import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
from urllib.parse import urlparse


# -----------------------------------
# Streamlit App Setup
# -----------------------------------
st.set_page_config(page_title="Article to Podcast Agent")
st.title("🎙️ Article to Podcast Agent")


# -----------------------------------
# Sidebar: API Keys Input
# -----------------------------------
st.sidebar.header("🔐 Enter Your API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")

keys_provided = all([openai_api_key, elevenlabs_api_key, firecrawl_api_key])

if not keys_provided:
    st.warning("Please provide all three API keys in the sidebar.")

# -----------------------------------
# Input: URL to Convert
# -----------------------------------
url = st.text_input("📄 Enter the URL of the article or blog post:", "")
generate_button = st.button("🎧 Generate Podcast", disabled=not keys_provided)


# -----------------------------------
# Helper: Validate URL
# -----------------------------------
def is_valid_url(input_url):
    parsed = urlparse(input_url)
    return all([parsed.scheme, parsed.netloc])


# -----------------------------------
# Main Action
# -----------------------------------
if generate_button:
    if not is_valid_url(url):
        st.warning("Please enter a valid URL that starts with http:// or https://")
    else:
        # Set environment variables for APIs
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

        with st.spinner("⏳ Processing: scraping, summarizing, and generating audio..."):
            try:
                # Define the agent with instructions
                blog_to_podcast_agent = Agent(
                    name="Suka blog to podcast",
                    agent_id="blog_to_podcast",
                    model=OpenAIChat(id="gpt-3.5-turbo"),
                    tools=[
                        ElevenLabsTools(
                            voice_id="21m00Tcm4TlvDq8ikWAM",
                            model_id="eleven_monolingual_v1",
                            target_directory="audio_generations"
                        ),
                        FirecrawlTools()
                    ],
                    description="You are an AI agent that can generate audio from blog articles.",
                    instructions=[
                        "When the user provides a blog post URL:",
                        "1. Use FireCrawlTools to scrape the blog content.",
                        "2. Create a concise summary of the blog content no longer than 2000 characters.",
                        "3. The summary should capture the key ideas and be engaging.",
                        "4. Use the ElevenLabsTools to convert the summary to audio.",
                        "5. Make sure the summary fits within the ElevenLabs API limit."
                    ],
                    markdown=True,
                    debug_mode=True
                )

                # Run the agent
                podcast: RunResponse = blog_to_podcast_agent.run(
                    f"Convert the blog to a podcast: {url}"
                )

                # Save audio
                save_dir = "audio_generations"
                os.makedirs(save_dir, exist_ok=True)

                if podcast.audio and len(podcast.audio) > 0:
                    filename = f"{save_dir}/podcast_{uuid.uuid4()}.wav"
                    write_audio_to_file(
                        audio=podcast.audio[0].base64_audio,
                        filename=filename
                    )

                    st.success("✅ Podcast generated successfully!")
                    audio_bytes = open(filename, "rb").read()

                    st.audio(audio_bytes, format="audio/wav")

                    st.download_button(
                        label="⬇️ Download Podcast",
                        data=audio_bytes,
                        file_name="podcast.wav",
                        mime="audio/wav"
                    )

                    if podcast.output:
                        st.markdown("### 📝 Summary Used for Podcast")
                        st.text_area("Summary", podcast.output, height=250)
                        st.download_button("📄 Download Summary", podcast.output, file_name="summary.txt")

                else:
                    st.error("❌ No audio was generated. Please try again later.")
            except Exception as e:
                logger.exception("Error during podcast generation")
                st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
