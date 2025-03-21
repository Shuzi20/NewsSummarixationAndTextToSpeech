import streamlit as st
from utils import get_news_articles, comparative_analysis, extract_topics, text_to_speech_hindi, translate_to_hindi
import os

st.title("📰 News Summarizer with Sentiment Analysis & TTS")

# Initialize session state for audio and articles
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None

if "articles" not in st.session_state:
    st.session_state["articles"] = []

# Input for company name and number of articles
company = st.text_input("Enter Company Name")
num_articles = st.slider("Select Number of Articles", 10, 50, 20)

if st.button("Fetch News"):
    articles = get_news_articles(company, num_articles)

    if articles:
        st.session_state["articles"] = articles
        st.write(f"✅ Extracted {len(articles)} articles for {company}")

        # Construct the output JSON structure
        output = {
            "Company": company,
            "Articles": [],
            "Comparative Sentiment Score": {}
        }

        positive_articles = []
        negative_articles = []
        neutral_articles = []

        # Build the articles section
        for idx, article in enumerate(articles):
            article_data = {
                "Title": article.get("title", "No Title"),
                "Summary": article.get("summary", "No summary available"),
                "Sentiment": article.get("sentiment", "Neutral"),
                "Topics": article.get("topics", [])
            }

            output["Articles"].append(article_data)

            # Categorize by sentiment
            sentiment = article.get("sentiment", "Neutral").lower()
            if sentiment == "positive":
                positive_articles.append({"Title": article["title"]})
            elif sentiment == "negative":
                negative_articles.append({"Title": article["title"]})
            else:
                neutral_articles.append({"Title": article["title"]})

        # Add Comparative Sentiment Score section
        comparison = comparative_analysis(articles)
        output["Comparative Sentiment Score"] = {
            "Sentiment Distribution": {
                "Positive": len(positive_articles),
                "Neutral": len(neutral_articles),
                "Negative": len(negative_articles)
            },
            "Positive Articles": positive_articles,
            "Negative Articles": negative_articles,
            "Coverage Differences": comparison.get("Coverage Differences", [])
        }

        # Display the structured JSON output
        st.json(output)

        # Automatically generate audio summary
        with st.spinner("Generating Hindi Summary Audio..."):
            summary_text = "\n".join([article.get("summary", "") for article in articles if "summary" in article])

            # Translate to Hindi
            hindi_summary = translate_to_hindi(summary_text)

            # Generate TTS using Coqui and store the audio file path in session state
            audio_filename = text_to_speech_hindi(
                st.session_state["articles"],
                comparison,
                company
            )

            if audio_filename:
                st.session_state["audio_file"] = audio_filename
                st.success("✅ Audio summary generated successfully!")
            else:
                st.error("❌ Failed to generate Hindi audio.")

# Display the audio file and download button if it exists
if st.session_state["audio_file"]:
    audio_file = st.session_state["audio_file"]

    # Display audio player (Coqui generates WAV files)
    st.audio(audio_file, format="audio/wav")

    # Create download link
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    st.download_button(
        label="📥 Download Hindi Audio",
        data=audio_bytes,
        file_name=os.path.basename(audio_file),
        mime="audio/wav"
    )
