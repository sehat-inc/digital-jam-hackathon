import fitz
import time
import re
import logging
import tempfile
from pathlib import Path
from typing import List, Set, Union

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFHighlighter:
    """Unified class for processing and highlighting PDFs based on semantic similarity."""

    def __init__(self, model, stopwords_set, similarity_threshold=0.63, min_sentence_length=10):
        """
        Initialize with external dependencies injected.
        
        :param model: A pre-initialized sentence transformer model with encode method
        :param stopwords_set: Set of stopwords
        :param similarity_threshold: Threshold for highlighting sentences
        :param min_sentence_length: Minimum length to consider a valid sentence
        """
        self.model = model
        self.stopwords = stopwords_set
        self.similarity_threshold = similarity_threshold
        self.min_sentence_length = min_sentence_length
        
        # Define legal important words
        self.legal_important_words: Set[str] = {
            'shall', 'must', 'will', 'not', 'no', 'nor', 'any', 'all', 'none',
            'may', 'might', 'can', 'cannot', 'should', 'would', 'hereby'
        }
        # Preserve legal terms by removing them from stopwords
        self.stopwords -= self.legal_important_words
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text while preserving legal terms."""
        try:
            text = text.lower()
            # Retain word characters, whitespace, and periods
            text = re.sub(r'[^\w\s.]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            words = text.split()
            # Remove stopwords unless they are legal important words
            words = [word for word in words if (word not in self.stopwords or word in self.legal_important_words)]
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            raise RuntimeError(f"Text preprocessing failed: {str(e)}")

    def _open_pdf(self, pdf_input: Union[str, bytes]) -> fitz.Document:
        """
        Open a PDF document.
        :param pdf_input: A file path (str) or PDF bytes.
        :return: A fitz.Document instance.
        """
        try:
            if isinstance(pdf_input, bytes):
                return fitz.open(stream=pdf_input, filetype="pdf")
            else:
                return fitz.open(pdf_input)
        except Exception as e:
            logger.error(f"PDF opening failed: {str(e)}")
            raise RuntimeError(f"Failed to open PDF: {str(e)}")

    def extract_text_from_pdf(self, pdf_input: Union[str, bytes]) -> str:
        """Extract text from a PDF safely."""
        try:
            with self._open_pdf(pdf_input) as doc:
                text = " ".join(page.get_text() for page in doc)
                logger.debug(f"Extracted text length: {len(text)}")
                return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation and newline heuristics.
        Only returns sentences longer than the minimum sentence length.
        """
        # List of sentence terminators
        terminators = ['。', '？', '!', '\n']
        # Normalize double newlines and duplicate terminators
        text = text.replace('\n\n', '。').replace('。。', '。')
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in terminators:
                sentence = ''.join(current).strip()
                if sentence and len(sentence) > self.min_sentence_length:
                    sentences.append(sentence)
                current = []
                
        # Add any trailing text as a sentence
        if current:
            sentence = ''.join(current).strip()
            if sentence and len(sentence) > self.min_sentence_length:
                sentences.append(sentence)
                
        logger.debug(f"Total sentences split: {len(sentences)}")
        return sentences

    def compute_semantic_similarity(self, summary: str, sentences: List[str]) -> NDArray:
        """Compute semantic similarity between a summary and each sentence."""
        try:
            # Break the summary into sentences
            summary_sentences = self.split_into_sentences(summary) if isinstance(summary, str) else summary
            
            # Preprocess both summary sentences and document sentences
            summary_sentences_proc = [self.preprocess_text(sent) for sent in summary_sentences]
            processed_sentences = [self.preprocess_text(sent) for sent in sentences]

            # Get embeddings
            summary_embeddings = self.model.encode(
                summary_sentences_proc,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            sentence_embeddings = self.model.encode(
                processed_sentences,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            # Convert tensor embeddings to numpy arrays
            summary_embeddings = summary_embeddings.cpu().numpy()
            sentence_embeddings = sentence_embeddings.cpu().numpy()

            # Weight summary sentences by their length
            weights = np.array([len(sent.split()) for sent in summary_sentences_proc], dtype=np.float32)
            weights /= weights.sum()
            
            # Compute a weighted average embedding for the summary
            avg_summary_embedding = np.average(summary_embeddings, axis=0, weights=weights)
            avg_summary_embedding /= np.linalg.norm(avg_summary_embedding)

            # Compute cosine similarity (dot product, since embeddings are normalized)
            similarities = np.dot(sentence_embeddings, avg_summary_embedding)
            
            # Apply a sigmoid to scale similarity scores between 0 and 1
            scaled_similarities = 1 / (1 + np.exp(-5 * (similarities - 0.5)))
            return scaled_similarities
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise RuntimeError(f"Similarity computation failed: {str(e)}")

    def highlight_pdf(self, pdf_input: Union[str, bytes], phrases: List[str]) -> bytes:
        """
        Add highlight annotations to the PDF for each phrase.
        Returns the highlighted PDF as bytes.
        """
        try:
            # Open the input PDF
            doc = self._open_pdf(pdf_input)
            
            # Iterate through each page and highlight found phrases
            for page in doc:
                for phrase in phrases:
                    # Search for the phrase on the page
                    for inst in page.search_for(phrase):
                        page.add_highlight_annot(inst)
                        
            # Save the updated PDF to a temporary file, then read its bytes
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                temp_filename = tmp_file.name
                
            try:
                doc.save(temp_filename)
                with open(temp_filename, "rb") as f:
                    highlighted_pdf_bytes = f.read()
            finally:
                Path(temp_filename).unlink(missing_ok=True)
                
            logger.debug("Highlighted PDF generated successfully.")
            return highlighted_pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF highlighting failed: {str(e)}")
            raise RuntimeError(f"Failed to highlight PDF: {str(e)}")

    def process_document(self, pdf_input: Union[str, bytes], summary: str) -> bytes:
        """
        Process a PDF document to highlight key sentences based on a summary.
        
        :param pdf_input: Either a file path (str) to a PDF or PDF data as bytes.
        :param summary: The summary text based on which sentences are highlighted.
        :return: The highlighted PDF as bytes.
        """
        start_time = time.time()
        logger.info("Starting document processing.")

        try:
            # Extract text from the PDF
            text = self.extract_text_from_pdf(pdf_input)
            
            # Split extracted text into sentences
            sentences = self.split_into_sentences(text)
            
            # Compute semantic similarity between the summary and each sentence
            similarities = self.compute_semantic_similarity(summary, sentences)
            
            # Select sentences that have a similarity score above the threshold
            important_phrases = [
                sent for sent, score in zip(sentences, similarities)
                if score > self.similarity_threshold
            ]
            
            logger.info(f"Found {len(important_phrases)} important phrase(s) to highlight.")
            
            # Highlight the important phrases in the PDF
            highlighted_pdf = self.highlight_pdf(pdf_input, important_phrases)
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return highlighted_pdf
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}")