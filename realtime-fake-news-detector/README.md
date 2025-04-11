# Real-Time Fake News Detector

A Python-based web application that detects fake news in real-time using machine learning techniques. The application provides instant feedback as users type or paste news articles, making it easy to verify the authenticity of news content quickly.

## Features

- Real-time news analysis using machine learning
- Live feedback as you type
- Detailed analysis results with confidence scores
- Key feature highlighting
- Text characteristics analysis
- User feedback system for continuous improvement
- Integration with NewsAPI for recent news
- Modern, responsive UI using Tailwind CSS

## Tech Stack

- **Backend**: Python, Flask, Flask-SocketIO
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Machine Learning**: scikit-learn, NLTK
- **Real-time Updates**: WebSocket
- **News API Integration**: NewsAPI

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- NewsAPI key (get one at [https://newsapi.org](https://newsapi.org))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd realtime-fake-news-detector
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your NewsAPI key:
   ```
   NEWS_API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage

1. **Analyze News**:
   - Paste or type news article text in the input area
   - Receive real-time analysis feedback
   - View detailed results including confidence scores and key features

2. **Recent News**:
   - Browse recent news articles from reliable sources
   - Click to analyze any article directly

3. **Provide Feedback**:
   - Help improve the system by providing feedback on analysis results
   - Mark results as helpful or not helpful

## Project Structure

```
realtime-fake-news-detector/
├── src/
│   ├── __init__.py
│   ├── news_fetcher.py    # News API integration
│   ├── preprocessor.py    # Text preprocessing
│   └── model.py          # ML model implementation
├── templates/
│   ├── index.html        # Main page
│   ├── result.html       # Analysis results
│   └── error.html        # Error page
├── app.py               # Flask application
├── requirements.txt     # Project dependencies
└── README.md           # Documentation
```

## Features in Detail

### Real-time Analysis
- Instant feedback as users type
- WebSocket connection for live updates
- Debounced input handling for optimal performance

### Text Analysis
- Preprocessing using NLTK
- TF-IDF vectorization
- PassiveAggressiveClassifier for classification
- Confidence score calculation
- Key feature extraction

### User Interface
- Clean, modern design with Tailwind CSS
- Responsive layout for all devices
- Interactive elements and animations
- Clear visualization of results

### Error Handling
- Comprehensive error catching
- User-friendly error messages
- Graceful fallback mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NewsAPI](https://newsapi.org) for providing news data
- [Tailwind CSS](https://tailwindcss.com) for the UI framework
- [Flask](https://flask.palletsprojects.com) for the web framework
- [scikit-learn](https://scikit-learn.org) for machine learning tools
