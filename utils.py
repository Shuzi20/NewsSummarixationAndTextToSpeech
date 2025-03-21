import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
from gtts import gTTS
from googletrans import Translator
import os
import pandas as pd
import nltk
import random
import time
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer

# =============== MODEL INITIALIZATION ===============
# Initialize the KeyBERT model at the top
kw_model = KeyBERT()

sia = SentimentIntensityAnalyzer()


# NLTK Downloads
nltk.download('vader_lexicon')
nltk.download('punkt')

# =============== HEADERS & CONFIGURATION ===============
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}
TIMEOUT = 15

def clean_text(text):
    """Removes unwanted phrases and unnecessary content from extracted text."""
    unwanted_phrases = [
        "Conditions of Use", "Privacy Policy", "© 1996-2025", "All Rights Reserved", 
        "Advertisement", "Skip to Main Content", "Trending news", "Oops, something went wrong"
    ]
    
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    
    # Remove multiple spaces and empty lines
    text = " ".join(text.split())
    
    return text.strip()

# =============== ARTICLE FETCHING WITH FALLBACKS ===============
def fetch_article_content(url):
    """Fetch and clean the article content for better summarization."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')

        clean_content = []
        for p in paragraphs:
            text = clean_text(p.get_text(strip=True))
            if len(text) > 20:  # Ensure meaningful text
                clean_content.append(text)

        content = " ".join(clean_content)
        if len(content) < 50:
            print(f"Insufficient content: {url}")
            return None

        summary = summarize_text(content)  # Call summarization function
        return {"content": content, "summary": summary}

    except Exception as e:
        print(f"Error fetching article: {e}")
        return None

# =============== NEWS SCRAPERS WITH FLEXIBILITY ===============
def scrape_duckduckgo(query, num_articles=10):
    """Scrape DuckDuckGo for news articles with flexible selectors."""
    url = f"https://duckduckgo.com/html/?q={query}+news"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = soup.select('.result__title') or soup.select('.result__a')

        news_data = []
        for article in articles[:num_articles]:
            title = article.text.strip()
            link = article.find('a')['href'] if article.find('a') else "No link"
            
            # Fetch article content
            content = fetch_article_content(link)
            if content:
                news_data.append({
                    'title': title,
                    'link': link,
                    'content': content
                })

        return news_data

    except Exception as e:
        print(f"Error in DuckDuckGo scraper: {e}")
        return []


def scrape_bing(query, num_articles=10):
    """Scrape Bing News with flexible selectors."""
    url = f"https://www.bing.com/news/search?q={query}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = soup.select('.news-card') or soup.select('.title')

        news_data = []
        for article in articles[:num_articles]:
            title = article.text.strip() if article else "No title"
            link_element = article.find('a')
            link = link_element['href'] if link_element else "No link"

            content = fetch_article_content(link)
            if content:
                news_data.append({
                    'title': title,
                    'link': link,
                    'content': content
                })

        return news_data

    except Exception as e:
        print(f"Error in Bing scraper: {e}")
        return []


def scrape_yahoo(query, num_articles=10):
    """Scrape Yahoo News with fallbacks."""
    url = f"https://news.search.yahoo.com/search?p={query}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = soup.select('div.NewsArticle') or soup.select('div.d-ib')

        news_data = []
        for article in articles[:num_articles]:
            title_element = article.select_one('h4') or article.select_one('h3')
            link_element = article.find('a')
            
            title = title_element.text.strip() if title_element else "No title"
            link = link_element['href'] if link_element else "No link"

            content = fetch_article_content(link)
            if content:
                news_data.append({
                    'title': title,
                    'link': link,
                    'content': content
                })

        return news_data

    except Exception as e:
        print(f"Error in Yahoo scraper: {e}")
        return []


# =============== AGGREGATOR FUNCTION ===============
def get_news_articles(query, min_articles=10, max_articles=30):
    """Aggregate articles from multiple search engines with summaries, sentiment, and topics."""
    all_articles = []
    unique_articles = set()

    engines = [scrape_duckduckgo, scrape_bing, scrape_yahoo]

    while len(all_articles) < min_articles:
        for engine in engines:
            try:
                articles = engine(query, max_articles // len(engines))

                for article in articles:
                    unique_id = f"{article['title']}|{article['link']}"
                    if unique_id not in unique_articles:
                        unique_articles.add(unique_id)

                        # Add summary, sentiment, and topics
                        article_content = fetch_article_content(article['link'])
                        
                        if article_content:
                            summary = article_content.get("summary", "No summary available")
                            sentiment = analyze_sentiment(summary)
                            topics = extract_topics([{"summary": summary}])[0]

                            article.update({
                                "summary": summary,
                                "sentiment": sentiment,
                                "topics": topics
                            })

                            all_articles.append(article)

                # Random delay to avoid getting blocked
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"Error with {engine.__name__}: {e}")

            if len(all_articles) >= min_articles:
                break

    print(f"Extracted {len(all_articles)} unique articles.")
    return all_articles


# =============== TRANSLATION FUNCTION ===============
def translate_to_hindi(text):
    """Translate English text to Hindi."""
    translator = Translator()
    try:
        translated = translator.translate(text, src='en', dest='hi')
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text


# =============== TEXT SUMMARIZATION ===============
def summarize_text(text, max_sentences=2):
    """Summarizes the given text by extracting key sentences."""
    if text:
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:max_sentences])
        return summary if summary else "No summary available"
    return "No summary available"



# =============== SENTIMENT ANALYSIS ===============
def analyze_sentiment(text):
    """Perform sentiment analysis on a given text."""
    if not text or text.strip() == "No summary available":
        return "Neutral"

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)

    if scores['compound'] > 0.05:
        return "Positive"
    elif scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# =============== COMPARATIVE ANALYSIS ===============
# =============== COMPARATIVE ANALYSIS ===============
def comparative_analysis(articles):
    """Conduct comparative sentiment & coverage analysis with impact statements."""
    
    extracted_articles = [a for a in articles if a.get("content")]
    
    if not extracted_articles:
        print("No valid articles with content for analysis.")
        return {}

    df = pd.DataFrame({
        "Title": [a["title"] for a in extracted_articles],
        "Content": [a["content"]["content"] for a in extracted_articles],
        "Summary": [a["content"]["summary"] for a in extracted_articles],
        "Sentiment": [analyze_sentiment(a["content"]["summary"]) for a in extracted_articles],
        "Topics": [extract_topics([{"summary": a["content"]["summary"]}])[0] for a in extracted_articles]
    })

    # Sentiment Distribution
    sentiment_distribution = df["Sentiment"].value_counts().to_dict()
    
    total_articles = len(df)
    positive_count = sentiment_distribution.get("Positive", 0)
    negative_count = sentiment_distribution.get("Negative", 0)
    neutral_count = sentiment_distribution.get("Neutral", 0)

    # Impact Analysis Based on Sentiment
    impact_statement = ""
    if positive_count > negative_count:
        impact_statement = "The overall media sentiment is **favorable**, indicating positive coverage and reputation."
    elif negative_count > positive_count:
        impact_statement = "The overall media sentiment is **negative**, suggesting potential reputational risks or concerns."
    else:
        impact_statement = "The sentiment is **balanced**, meaning media coverage is neutral or mixed."

    # TF-IDF Similarity for Coverage Differences
    contents = df["Content"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Grouping articles with high similarity
    groups = []
    used = set()

    for i in range(len(extracted_articles)):
        if i in used:
            continue

        group = [df.iloc[i]["Title"]]
        used.add(i)

        for j in range(i + 1, len(extracted_articles)):
            if j not in used and similarity_matrix[i, j] > 0.5:  # Threshold for similarity
                group.append(df.iloc[j]["Title"])
                used.add(j)

        groups.append(group)

    # Extract topic overlap
    topic_overlap = {"Common Topics": set(), "Unique Topics": {}}
    for group in groups:
        common_topics = set(df[df["Title"].isin(group)]["Topics"].sum())
        topic_overlap["Common Topics"].update(common_topics)

        for title in group:
            article_topics = set(df[df["Title"] == title]["Topics"].sum())
            topic_overlap["Unique Topics"][title] = list(article_topics - common_topics)

    # Coverage Differences Summary
    coverage_differences = []
    for group in groups:
        if len(group) > 1:
            coverage_differences.append({
                "Similar Articles": group,
                "Shared Topics": list(topic_overlap["Common Topics"])
            })
        else:
            coverage_differences.append({
                "Unique Article": group[0],
                "Unique Topics": topic_overlap["Unique Topics"].get(group[0], [])
            })

    # **Final Impact Statement on Coverage Differences**
    if len(coverage_differences) > 5:
        coverage_impact = "The news coverage is **diverse**, indicating broad topic exploration across multiple sources."
    elif len(coverage_differences) <= 2:
        coverage_impact = "The news coverage is **highly similar**, with many sources reporting the same key points."
    else:
        coverage_impact = "The news coverage is **moderately varied**, balancing repeated narratives and unique insights."

    return {
        "Sentiment Distribution": sentiment_distribution,
        "Impact Statement": impact_statement,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": topic_overlap,
        "Coverage Impact": coverage_impact
    }

# =============== TOPIC EXTRACTION ===============
def extract_topics(articles):
    """Extract topics using KeyBERT with better handling for missing summaries."""
    topics = []
    for article in articles:
        summary = article.get('summary', '').strip()
        
        if summary and summary != "No summary available":
            try:
                keywords = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), top_n=5)
                topics.append([kw[0] for kw in keywords] if keywords else ["No topics available"])
            except Exception as e:
                print(f"Error extracting topics: {e}")
                topics.append(["No topics available"])
        else:
            topics.append(["No topics available"])
    
    return topics


# =============== HINDI TEXT-TO-SPEECH (REPORT) ===============
def text_to_speech_hindi(articles, comparison, company_name):
    """Generate a consolidated sentiment report in Hindi speech using gTTS."""
    try:
        # Extracting key details
        total_articles = len(articles)
        sentiment_dist = comparison["Sentiment Distribution"]
        positive_count = sentiment_dist.get("Positive", 0)
        negative_count = sentiment_dist.get("Negative", 0)
        neutral_count = sentiment_dist.get("Neutral", 0)

        # Constructing the report
        report = (
            f"कंपनी {company_name} के लिए कुल {total_articles} समाचार लेखों का विश्लेषण किया गया। "
            f"इनमें {positive_count} सकारात्मक, {negative_count} नकारात्मक और {neutral_count} तटस्थ लेख पाए गए।"
        )

        # Generating speech using gTTS
        tts = gTTS(text=report, lang='hi', slow=False)

        # Define output filename
        filename = f"{company_name}_report.mp3"
        output_path = os.path.join("output", filename)
        os.makedirs("output", exist_ok=True)

        # Save the audio file
        tts.save(output_path)

        print(f"TTS report saved successfully at {output_path}")
        return output_path

    except Exception as e:
        print(f"Error in Hindi TTS report generation: {e}")
        return None