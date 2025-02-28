
<img src="assests/logo.png" alt="Covenant AI Logo" width="150" height="150">

<div align="left">
   <a href="https://sehat-inc.github.io/covenant-ai/">
      <img src="https://img.shields.io/badge/Docs-Covenant%20AI%20User%20Guide-blueviolet?style=flat&logo=read-the-docs" alt="Covenant AI User Guide">
   </a>
   <a href="https://www.bestpractices.dev/projects/9976">
      <img src="https://www.bestpractices.dev/projects/9976/badge" alt="Covenant AI Best Practices">
   </a>
</div>

## Covenant AI

Covenant AI is a smart lease contract analysis system that leverages Agentic RAG for information retrieval.
The RAG architecture is built without the use of external libraries/frameworks for scalability. 

> For a comprehensive overview, please visit [the offical Covenant AI Documentation](https://docs.google.com/document/d/1wF_BIUqBmTT5pCod0L1EdVWjJe7RHlbvKfGV_hV0Fkc/edit?usp=sharing) 

## Sample Dataset for Hackathon

The project root has a few sample lease agreements you can use to test the project and its funcationalities.

## Features

### Frontend (HTML/CSS/JavaScript)
- Contract Upload: Users can upload lease contracts in PDF or DOCX format.
- Summary & Insights: Provides an AI-generated summary of key terms.
- Highlight PDF: Highlights the important clauses and obligations in the pdf.
- Interactive Chatbot: Allows users to ask AI-driven questions about the contract.

### Backend (Flask Framework)
- Document Processing: Extracts and parses text from uploaded documents using OCR.
- NLP Analysis: Uses AI models to analyze legal language.
- Database Management: Stores analyzed contracts in the supabase.
- Dynamic ground rules: Update rules and regulations that must be present in agreement. 

### AI Components
1. Summarization: Summarize the uploaded agreements with LLM.
2. Compare agreements: Compare 2 agreements and see which one is better.
3. ChatBot: RAG based chatbot that helps understand jargons.
4. Multiple contracts: Upload multiple PDFs and can chat with any of them.

## Technologies Used
- Frontend: HTML, CSS, JavaScript
- Backend: Flask Framework, SupaBase
- AI & NLP: SpaCy, Transformers (BERT/GPT), OCR for text extraction
- Cloud & APIs: AWS S3 for file storage, FastAPI for AI models


## 📜 Get Started


### **Prerequisites**
- Python 3.x
- SupaBase
- Flask framework
- NLTK, Sentence Transformers
- HTML, CSS, JS


### Installing requirements
`pip install -r requirements.txt`

### Installing Tesseract
Below are platform-specific instructions:

### Windows
1. Download and install from:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Ensure the tesseract.exe location is added to your PATH.

### macOS
1. Install Homebrew if needed (https://brew.sh).
2. Run: `brew install tesseract`.

### Linux (Ubuntu/Debian)
1. Update your package list: `sudo apt-get update`.
2. Install Tesseract: `sudo apt-get install tesseract-ocr`.

After installation, verify with:  
`tesseract --version`


## Running project
**root/src**:
                  ` python app.py`
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For inquiries, contact u2022378@giki.edu.pk
