from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk

from highlight_key_terms import PDFHighlighter

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize SentenceTransformer model - will be shared across requests
model = SentenceTransformer('all-MiniLM-L6-v2')
stopwords_set = set(stopwords.words('english'))

# Create a single instance of PDFHighlighter
pdf_highlighter = PDFHighlighter(
    model=model,
    stopwords_set=stopwords_set,
    similarity_threshold=0.63,
    min_sentence_length=10
)

# Test the PDF highlighter
pdf_path = r"C:\Users\mh407\OneDrive\Documents\AI Projects\ai-lease proj\covenant-ai\rag\data\raw\testPDF.pdf"  # Replace with your PDF file path
summary = """
**Lease Contract Summary**

This summary outlines key aspects of the lease contract, aiming for clarity and comprehensiveness.  Due to the highly technical nature of the original document, some legal nuances may require further professional consultation.

**I. Depreciation Charges:**

* **Program Vehicles (within Estimation Period):** Depreciation is calculated based on the "Initially Estimated Depreciation Charge" as of the relevant date.
* **Program Vehicles (outside Estimation Period):** Depreciation uses the monthly dollar amount specified in the related Manufacturer Program.
* **Non-Program Medium-Duty Trucks:** Depreciation is calculated according to Generally Accepted Accounting Principles (GAAP) and uses a percentage based on the truck's age:
    * 0-12 months: 2.75%
    * 13-24 months: 1.42%
    * >24 months: 0.58%
* **Depreciation Record:**  The specific definition is detailed in Section 4.1 of the Lease.

**II. Key Dates and Definitions:**

* **Determination Date:** Five business days before each Payment Date.
* **Direct-to-Consumer Sale:**  A sale where Hertz or its affiliate transfers vehicle title and acts as the seller, complying with consumer protection laws.
* **Disposition Date:** The date determining the vehicle's value, varying based on the circumstances:
    * **Manufacturer Repurchase:**  The "Turnback Date."
    * **Guaranteed Depreciation Program (not sold to third party):** The "Backstop Date."
    * **Sold to Third Party:** The date sale proceeds are deposited into the "Collection Account."

**III. Special Terms:**

The lease includes "Special Terms"  that vary by vehicle location, defining the lease's duration in specific states/commonwealths.

* **Illinois:** One year
* **Iowa:** Eleven months
* **Maine:** Eleven months
* **Maryland:** 180 days
* **Massachusetts:** Eleven months
* **Nebraska:** Thirty days
* **South Dakota:** Twenty-eight days
* **Texas:** 181 days
* **Vermont:** Eleven months
* **Virginia:** Eleven months
* **West Virginia:** Thirty days


**IV.  Other Key Terms & Definitions (Summary):**

* **Required Series Noteholders:** Defined in the Base Indenture.
* **Resigning Lessee:** Defined in Section 25 of the Lease.
* **SEC:** Securities and Exchange Commission.
* **Series of Notes:** Notes issued under the Base Indenture and Series Supplement.
* **Series Supplement:** Supplement to the Base Indenture.
* **Servicer:** Defined in the Lease Preamble.
* **Servicer Default:** Defined in Section 9.6 of the Lease.
* **Servicing Standard:**  Describes the expected level of service, emphasizing promptness, diligence, and industry best practices.  Failure to meet this standard must not materially adversely affect the Lessor.

**V. Missing Information:**

This summary is based solely on the provided text.  Crucial information such as lease duration (outside of the special terms), payment amounts, penalties for late payments, lessee obligations, termination clauses (besides the definition of "Resigning Lessee"), and renewal terms are not included in the extract and therefore cannot be summarized.  A complete review of the entire lease agreement is necessary to obtain this information.

"""

try:
    # Process the document and get highlighted PDF bytes
    highlighted_pdf_bytes = pdf_highlighter.process_document(pdf_path, summary)
    
    # Save the highlighted PDF
    output_path = "sample.pdf"
    with open(output_path, "wb") as f:
        f.write(highlighted_pdf_bytes)
    print(f"Successfully processed and saved highlighted PDF to {output_path}")
    
except Exception as e:
    print(f"Error processing PDF: {str(e)}")

