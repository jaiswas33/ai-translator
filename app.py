import os
import logging
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from vertexai.generative_models import GenerativeModel
import vertexai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import fitz  # PyMuPDF for PDF text extraction
import textwrap

# --- Config ---
PROJECT_ID = os.getenv("PROJECT_ID", "eight-brothers")
REGION = os.getenv("REGION", "us-central1")
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

# --- PDF Output Utility with Unicode Support ---
class TranslatedPDF:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter,
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=18)
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Create a custom style for better text handling
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            wordWrap='LTR'
        )
    
    def add_text_block(self, text):
        """Add text block to PDF with proper Unicode handling"""
        try:
            # Split text into paragraphs
            paragraphs = text.split('\n')
            
            for para in paragraphs:
                if para.strip():  # Skip empty paragraphs
                    # Handle very long lines by breaking them
                    if len(para) > 100:
                        wrapped_lines = textwrap.wrap(para, width=80)
                        for line in wrapped_lines:
                            p = Paragraph(line, self.normal_style)
                            self.story.append(p)
                    else:
                        p = Paragraph(para, self.normal_style)
                        self.story.append(p)
                else:
                    # Add space for empty paragraphs
                    self.story.append(Spacer(1, 6))
        except Exception as e:
            logging.error(f"Error adding text block: {e}")
            # Fallback: add as plain text with error handling
            fallback_text = f"Translation completed. Text encoding issues prevented proper formatting."
            p = Paragraph(fallback_text, self.normal_style)
            self.story.append(p)
    
    def save(self):
        """Save the PDF"""
        try:
            self.doc.build(self.story)
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            # Create a simple fallback PDF
            self._create_fallback_pdf()
    
    def _create_fallback_pdf(self):
        """Create a simple fallback PDF when Unicode issues occur"""
        try:
            c = canvas.Canvas(self.filename, pagesize=letter)
            width, height = letter
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height - 72, "Translation Complete")
            
            # Add message
            c.setFont("Helvetica", 12)
            message = "The translation was completed successfully."
            c.drawString(72, height - 120, message)
            
            message2 = "Due to character encoding limitations, the full text"
            c.drawString(72, height - 140, message2)
            
            message3 = "could not be displayed in this PDF format."
            c.drawString(72, height - 160, message3)
            
            message4 = "Please try downloading as a text file instead."
            c.drawString(72, height - 180, message4)
            
            c.save()
        except Exception as e:
            logging.error(f"Error creating fallback PDF: {e}")

# --- Alternative: Simple text file output ---
def create_text_file(text, filename):
    """Create a text file as alternative to PDF"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logging.error(f"Error creating text file: {e}")
        return False

# --- Vertex AI Setup ---
def initialize_vertex_ai():
    try:
        vertexai.init(project=PROJECT_ID, location=REGION)
        return GenerativeModel(MODEL_NAME).start_chat(history=[])
    except Exception as e:
        logging.error("Vertex AI initialization failed: %s", e)
        return None

chat = initialize_vertex_ai()

# --- Text Extraction ---
def extract_text(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    else:
        raise ValueError("Unsupported file type")

# --- Gemini Translation ---
def translate_text(text, target_language):
    if not chat:
        raise RuntimeError("Gemini not initialized")
        
    prompt = f"""
    Detect the language of the following content and translate it into {target_language}.
    Only return the translated version without commentary.
        
    ```
    {text}
    ```
    """
    response = chat.send_message(prompt)
    return response.text

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "Empty filename", 400

        target_language = request.form.get('language')
        if not target_language:
            return "No target language specified", 400
        
        output_format = request.form.get('format', 'pdf')  # Allow format selection

        # Save file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

        # Extract and translate
        original_text = extract_text(file_path)
        translated = translate_text(original_text, target_language)

        # Generate output based on format preference
        if output_format == 'txt':
            # Create text file
            output_file = "translated_output.txt"
            if create_text_file(translated, output_file):
                return send_file(output_file, as_attachment=True, 
                               download_name=f"translated_{target_language}.txt")
            else:
                return "Error creating text file", 500
        else:
            # Try to create PDF
            try:
                output_pdf = "translated_output.pdf"
                pdf = TranslatedPDF(output_pdf)
                pdf.add_text_block(translated)
                pdf.save()
                return send_file(output_pdf, as_attachment=True,
                               download_name=f"translated_{target_language}.pdf")
            except Exception as pdf_error:
                logging.error(f"PDF creation failed: {pdf_error}")
                # Fallback to text file
                output_file = "translated_output.txt"
                if create_text_file(translated, output_file):
                    return send_file(output_file, as_attachment=True,
                                   download_name=f"translated_{target_language}.txt")
                else:
                    return f"Error creating output file: {pdf_error}", 500

    except Exception as e:
        logging.exception("Error in translation")
        return f"Error: {e}", 500

# --- Health check route ---
@app.route('/health')
def health():
    return {"status": "healthy", "vertex_ai": chat is not None}

# --- Main ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)