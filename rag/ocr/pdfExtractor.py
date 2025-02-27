"""
Author: Hamza Amin
Description: Extract text and tables from PDF files using PyMuPDF and pdfplumber
Date: 2025-30-01

"""

import pdfplumber
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Data class to store table information"""
    content: List[List[str]]
    page_number: int
    location: Tuple[float, float, float, float]


class PDFTextExtractor:
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF text extractor

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to store output files
        """
        self.pdf_path = Path(pdf_path)
        self.extracted_text = []
        self.tables = []

    def _check_for_scanned_content(self, page) -> bool:
        """
        Check if the page might be scanned/image-based

        Args:
            page: PDF page object
        Returns:
            bool: True if page appears to be scanned
        """
        text = page.get_text()
        if not text or len(text.strip()) < 50:  # Arbitrary threshold
            return True
        return False

    def _process_scanned_page(self, page) -> str:
        """
        Process scanned pages using OCR

        Args:
            page: PDF page object
        Returns:
            str: Extracted text from the scanned page
        """
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng")
            return text
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""

    def _extract_tables(self, page) -> List[TableData]:
        """
        Extract tables from the page while preserving structure

        Args:
            page: PDFPlumber page object
        Returns:
            List[TableData]: List of extracted tables
        """
        tables = []
        try:
            for table in page.extract_tables():
                if table:
                    # Clean and process table data
                    processed_table = [
                        [str(cell).strip() if cell else "" for cell in row]
                        for row in table
                    ]
                    # Filter out empty rows and columns
                    processed_table = [
                        row for row in processed_table if any(cell for cell in row)
                    ]

                    if processed_table:
                        tables.append(
                            TableData(
                                content=processed_table,
                                page_number=page.page_number,
                                location=(
                                    0, 0, 0, 0,                                
                                ),  # Default bbox since pdfplumber page doesn't provide it directly
                            )
                        )
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")

        return tables

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace while preserving formatting
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def extract_text(self) -> Dict:
        """
        Main method to extract text and tables from PDF

        Returns:
            Dict: Extracted content including text and tables
        """
        try:
            # Use both PyMuPDF and pdfplumber for optimal extraction
            doc = fitz.open(self.pdf_path)

            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num in range(len(doc)):
                    # Get PyMuPDF page for text extraction
                    mupdf_page = doc[page_num]

                    # Get pdfplumber page for table extraction
                    plumber_page = pdf.pages[page_num]

                    # Extract text using PyMuPDF
                    if self._check_for_scanned_content(mupdf_page):
                        text = self._process_scanned_page(mupdf_page)
                    else:
                        text = mupdf_page.get_text()

                    # Extract tables using pdfplumber
                    page_tables = self._extract_tables(plumber_page)
                    self.tables.extend(page_tables)

                    # Clean and store text
                    cleaned_text = self._clean_text(text)
                    self.extracted_text.append(
                        {
                            "page_number": page_num + 1,
                            "text": cleaned_text,
                            "tables": [table.content for table in page_tables],
                        }
                    )

            doc.close()
            return {"text": self.extracted_text, "tables": self.tables}

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def save_to_file(self) -> None:
        """Save extracted content to files"""
        try:
            # Save text content
            output_text_path = self.output_dir / f"{self.pdf_path.stem}_extracted.txt"
            with open(output_text_path, "w", encoding="utf-8") as f:
                for page in self.extracted_text:
                    f.write(f"\n=== Page {page['page_number']} ===\n\n")
                    f.write(page["text"])
                    if page["tables"]:
                        f.write("\n\n=== Tables ===\n\n")
                        for table in page["tables"]:
                            df = pd.DataFrame(table)
                            f.write(df.to_string())
                            f.write("\n\n")

            logger.info(f"Extracted content saved to {output_text_path}")

        except Exception as e:
            logger.error(f"Error saving to file: {str(e)}")
            raise