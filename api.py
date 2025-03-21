from flask import Flask, jsonify, request
from utils import get_news_articles, comparative_analysis, extract_topics

app = Flask(__name__)

@app.route('/get_news', methods=['GET'])
def get_news():
    """API endpoint to fetch news and perform analysis."""
    company = request.args.get('company')
    articles = get_news_articles(company)

    if articles:
        comparison = comparative_analysis(articles)
        topics = extract_topics(articles)

        return jsonify({
            "company": company,
            "articles": articles,
            "comparison": comparison,
            "topics": topics
        })
    else:
        return jsonify({"error": "No articles found."})

if __name__ == '__main__':
    app.run(debug=True)

