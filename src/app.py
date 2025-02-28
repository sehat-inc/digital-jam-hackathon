import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from sentence_transformers import SentenceTransformer
from rag.core.chunking import SemanticChunker
from rag.ocr.pdfExtractor import PDFTextExtractor
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from supabase import create_client, Client
from rag.core.summarizer import SummarizerAgent
import google.generativeai as genai
import tempfile
from datetime import datetime
from rag.ocr.highlight_key_terms import PDFHighlighter
from dotenv import load_dotenv
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))


load_dotenv()   
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

supabase: Client = create_client(
    os.getenv('SERVICE_KEY'),
    os.getenv('ROLE_KEY')
)

BUCKET_NAME = 'contract-files'  # Changed bucket name to be more specific

# Initialize models and services only once at the start
print("Initializing SentenceTransformer - This should happen only once")
encoder = SentenceTransformer("all-MiniLM-L6-v2") 

# Initialize Gemini for summarization
genai.configure(api_key=os.getenv('GEMINI_API'))
model = genai.GenerativeModel("gemini-1.5-flash")
summarizer = SummarizerAgent(llm=model)

# Create chunker with the encoder
chunker = SemanticChunker(model=encoder, min_tokens=100, max_tokens=1024)

pdf_highlighter = PDFHighlighter(
    model=encoder,
    stopwords_set=stopwords_set,
    similarity_threshold=0.63,
    min_sentence_length=10
)
# Custom filter for datetime formatting
@app.template_filter('format_datetime')
def format_datetime(value):
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y %I:%M %p')
        except ValueError:
            return value
    return value

@app.route('/')
def index():
    # Fetch all contracts from Supabase
    response = supabase.table('Contract').select('*').order('created_at.desc').execute()
    contracts = response.data
    return render_template('index.html', contracts=contracts)