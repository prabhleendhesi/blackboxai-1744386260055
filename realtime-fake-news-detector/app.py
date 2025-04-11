from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from src.model import FakeNewsDetector
from src.news_fetcher import NewsAPI
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app)

# Initialize our detector and news API
detector = FakeNewsDetector()
news_api = NewsAPI(api_key=os.getenv('NEWS_API_KEY', ''))

@app.route('/')
def index():
    """Render the main page."""
    try:
        # Fetch some recent news for display
        recent_news = news_api.fetch_latest_news(page_size=5)
        return render_template('index.html', recent_news=recent_news)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error="Failed to load the page. Please try again.")

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the provided news text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Get prediction
        result = detector.predict(text)
        
        # Get text characteristics
        characteristics = detector.analyze_text_characteristics(text)
        
        # Combine results
        response = {
            'prediction': result,
            'characteristics': characteristics
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    """Fetch news articles based on topic or source."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        source = data.get('source')
        
        if topic:
            articles = news_api.get_news_by_topic(topic)
        elif source:
            articles = news_api.get_news_by_source(source)
        else:
            articles = news_api.fetch_latest_news()
            
        return jsonify({'articles': articles})
        
    except Exception as e:
        logger.error(f"Error in fetch-news route: {str(e)}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@socketio.on('analyze_text')
def handle_realtime_analysis(data):
    """Handle real-time text analysis through WebSocket."""
    try:
        text = data.get('text', '')
        if text:
            # Get prediction
            result = detector.predict(text)
            # Emit result back to client
            emit('analysis_result', result)
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        emit('analysis_error', {'error': 'Analysis failed'})

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Handle user feedback for improving the model."""
    try:
        data = request.get_json()
        text = data.get('text')
        label = data.get('label')  # 0 for real, 1 for fake
        
        if text is None or label is None:
            return jsonify({'error': 'Missing text or label'}), 400
            
        # Update model with feedback
        success = detector.update_model(text, label)
        
        if success:
            return jsonify({'message': 'Feedback recorded successfully'})
        else:
            return jsonify({'error': 'Failed to record feedback'}), 500
            
    except Exception as e:
        logger.error(f"Error in feedback route: {str(e)}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    try:
        # Ensure the model is initialized
        logger.info("Initializing the fake news detector...")
        
        # Start the Flask-SocketIO app
        port = int(os.getenv('PORT', 8000))
        logger.info(f"Starting server on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start the application: {str(e)}")
