import logging
import traceback
import markdown
from flask import Flask, request, jsonify  # Web app + handle HTTP requests
from openai import OpenAI  # To communicate with NVIDIA LLM API
from flask_cors import CORS  # Cross-origin requests
import pdfplumber  # Library to extract text from PDF
import pytesseract  # OCR for image-based PDFs
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
from PIL import Image  # To process images from PDFs
# Set up logging to capture detailed error messages
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app) #cross orgin resource sharing

# Configure NVIDIA's LLM API
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = "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"  # Replace with your NVIDIA API key


# Route for uploading a file and summarizing its content
@app.route('/summarize', methods=['POST'])
def summarize_file():
    try:
        logging.debug("Request received for /summarize endpoint")

        # Check if file is uploaded
        if 'file' not in request.files:
            logging.error("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400
        
        uploaded_file = request.files['file']
        logging.debug("File uploaded: %s", uploaded_file.filename)

        if uploaded_file.filename == '':
            logging.error("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Extract text from file
        text_data = ""
        if uploaded_file.filename.endswith('.pdf'):
            logging.debug("Processing PDF file")
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        # Attempt to extract text from the PDF page
                        page_text = page.extract_text()
                        if page_text:
                            text_data += page_text + "\n"
                        else:
                            # If no text found, process the page as an image for OCR
                            logging.debug("No text found, using OCR on the page")
                            page_image = page.to_image(resolution=300)
                            image = page_image.original
                            ocr_text = pytesseract.image_to_string(image)
                            text_data += ocr_text + "\n"
                logging.debug("Text extracted from PDF successfully")
            except Exception as e:
                logging.error("Error extracting text from PDF: %s", str(e))
                return jsonify({"error": "Error extracting text from PDF", "details": str(e)}), 500
        else:
            logging.debug("Processing plain text file")
            text_data = uploaded_file.read().decode('utf-8')
            logging.debug("Text extracted from plain text file successfully")

        # Get custom prompt from the request
        custom_prompt = request.form.get('prompt', 'Summarize the content.')  # Default prompt if none provided
        logging.debug("Custom prompt: %s", custom_prompt)

        # Call NVIDIA's API for summarization
        client = OpenAI(base_url=NVIDIA_API_BASE_URL, api_key=API_KEY)

        # Send the content to the NVIDIA LLM
        logging.debug("Calling NVIDIA's API")
        try:
            completion = client.chat.completions.create(
               model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": "summarize this"},
                    {"role": "user", "content": text_data}
                ],
                temperature=0.8,
                top_p=1,
                max_tokens=2048,
                stream=True
            )
            logging.debug("Model response received")
        except Exception as e:
            logging.error("General error while calling NVIDIA API: %s", str(e))
            return jsonify({"error": "An error occurred while calling NVIDIA API", "details": str(e)}), 500

        # Extract the summarized content
        summary = ""
        try:
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:   #This condition ensures that content exists before appending it to the summary
                    summary += chunk.choices[0].delta.content
            logging.debug("Summary generated: %s", summary)
        except Exception as e:
            logging.error("Error while extracting summary: %s", str(e))
            return jsonify({"error": "Error while extracting summary", "details": str(e)}), 500
        return jsonify({"summary": summary}), 200

    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace: %s", traceback.format_exc())  # Print stack trace for debugging
        return jsonify({"error": "An error occurred during summarization", "details": str(e)}), 500
    
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
