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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'contract' not in request.files:
        return redirect(request.url)
    
    file = request.files['contract']
    contract_title = request.form.get('contract_title')  # Get the title from form

    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Save file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            # Extract text using OCR
            extractor = PDFTextExtractor(temp_path)
            extracted_content = extractor.extract_text()
            print("Extraction done ", datetime.now().time())
            
            # Get text from all pages
            all_text = "\n".join([page['text'] for page in extracted_content['text']])
            
            # Use the pre-initialized chunker instead of creating a new one
            chunked_text = chunker.chunk_text(text=all_text)
            print("Chunking done: ", datetime.now().time())
           
            # Generate summary
            summary = summarizer._run(text=all_text)
            print("Summary made: ", datetime.now().time())
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{timestamp}_{secure_filename(file.filename)}"
            
            # Upload PDF to Supabase Storage
            with open(temp_path, 'rb') as f:
                file_data = f.read()
                result = supabase.storage.from_(BUCKET_NAME).upload(
                    path=file_name,
                    file=file_data,
                    file_options={"content-type": "application/pdf"}
                )
                print(f"Upload result: {result}")
            
            # Create database entry with contract title
            contract_data = {
                'title': contract_title,  # Adding title
                'created_at': datetime.now().isoformat(),
                'contract_pdf': file_name,
                'contract_summary': summary
            }
            
            insert_result = supabase.table('Contract').insert(contract_data).execute()
            print(f"Database insert result: {insert_result}")
            print(datetime.now().time())
            
            # Cleanup
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Error during upload: {str(e)}")
            # Cleanup on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return f"Error uploading file: {str(e)}", 500
    else:
        return "Invalid file type. Please upload a PDF.", 400



@app.route('/highlight_pdf/<int:id>', methods=['GET', 'POST'])
def highlight_pdf(id):
    # Fetch contract details from Supabase
    response = supabase.table('Contract').select('*').eq('id', id).execute()
    if not response.data:
        return "Contract not found", 404
    
    contract = response.data[0]

    # **Get summary from database**
    summary = contract.get('contract_summary', '')
    if not summary:
        return "No summary available for highlighting", 400  # Bad request

    # Handle POST request
    if request.method == "POST":
        try:
            # Check if highlighted PDF already exists
            if contract.get('highlight_pdf'):
                return redirect(url_for('view_highlighted_pdf', id=id))

            # Download the original PDF
            pdf_data = supabase.storage.from_(BUCKET_NAME).download(contract['contract_pdf'])

            # Generate highlighted PDF using the **database summary**
            highlighted_pdf_bytes = pdf_highlighter(pdf_data, summary)

            # Unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            highlighted_file_name = f"{timestamp}_highlighted_{contract['contract_pdf']}"

            # Upload highlighted PDF
            supabase.storage.from_(BUCKET_NAME).upload(
                path=highlighted_file_name,
                file=highlighted_pdf_bytes,
                file_options={"content-type": "application/pdf"}
            )

            # Update database
            supabase.table('Contract').update({
                'highlight_pdf': highlighted_file_name
            }).eq('id', id).execute()

            return redirect(url_for('view_highlighted_pdf', id=id))

        except Exception as e:
            print(f"Error during highlighting: {str(e)}")
            return f"Error highlighting file: {str(e)}", 500

    # **If GET request, just display the contract instead of redirecting infinitely**
    return render_template('contract.html', contract=contract)



@app.route('/highlighted_pdf/<int:id>')
def view_highlighted_pdf(id):
    # Fetch contract details from Supabase
    response = supabase.table('Contract').select('*').eq('id', id).execute()
    if not response.data:
        return "Contract not found", 404
    
    contract = response.data[0]
    
    # Get public URL for highlighted PDF
    filename = contract['highlight_pdf'].split('/')[-1]
    contract['highlight_pdf_url'] = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
    
    return render_template('highlight_pdf.html', contract=contract)

@app.route('/contract/<int:id>')
def view_contract(id):
    # Fetch contract details from Supabase
    response = supabase.table('Contract').select('*').eq('id', id).execute()
    if not response.data:
        return "Contract not found", 404
    
    contract = response.data[0]
    
    # Get public URL correctly
    # Remove any full URLs or double slashes from the filename
    filename = contract['contract_pdf'].split('/')[-1]
    contract['pdf_url'] = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
    
    return render_template('contract.html', contract=contract)


@app.route('/download/<int:contract_id>')
def download_contract(contract_id):
    try:
        # Fetch contract details from Supabase
        response = supabase.table('Contract').select('*').eq('id', contract_id).execute()
        if not response.data:
            return "Contract not found", 404
        
        contract = response.data[0]
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # Download the file data
        data = supabase.storage.from_(BUCKET_NAME).download(contract['contract_pdf'])
        
        # Write to temporary file
        with open(temp_file.name, 'wb') as f:
            f.write(data)
        
        return send_file(
            temp_file.name,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=os.path.basename(contract['contract_pdf'])
        )
    except Exception as e:
        print(f"Error downloading contract: {str(e)}")
        return f"Error downloading contract: {str(e)}", 500
    finally:
        # Cleanup temp file
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == '__main__':
    app.run(debug=True)


