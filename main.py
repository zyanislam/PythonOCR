import json
import os
import cv2
import pytesseract
import numpy as np

from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

pdfs_directory = "/home/quantum/Downloads/Projects/PythonOCR/pdfs"
output_folder = "images"

# Create the "images" folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all PDF files in the "pdfs" directory
pdf_files = [f for f in os.listdir(pdfs_directory) if f.endswith('.pdf')]

# Create a dictionary to store the OCR results and images for each PDF
pdf_ocr_results = {}

# Loop through each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdfs_directory, pdf_file)

    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Create a list to store images and text for the current PDF
    pdf_images = []
    result_dict = {}

    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(img_rgb)
        result_dict[f"page_{i+1}"] = {"text": text, "img_path": f"{pdf_file.replace('.pdf', '')}_page_pg{i+1}.png"}

        # Append the image to the list
        pdf_images.append(np.array(img))

    # Concatenate images vertically
    merged_image = cv2.vconcat(pdf_images)

    # Save the merged image to the "images" folder with the desired filename format without .pdf extension
    img_filename = f"{pdf_file.replace('.pdf', '')}_merged.png"
    cv2.imwrite(os.path.join(output_folder, img_filename), cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))

    # Combine text and image paths for the current PDF
    pdf_ocr_results[pdf_file] = {
        "text": "\n".join(result["text"] for result in result_dict.values()),
        "img_path": os.path.join(output_folder, img_filename),
        "matched_keywords": []
    }

# Convert the dictionary to JSON for all PDFs
json_data_all_pdfs = json.dumps(pdf_ocr_results, indent=2)

# Print or save the JSON data for all PDFs
print("JSON Data for All PDFs:")
print(json_data_all_pdfs)

# Prompt the user for keywords
user_keywords = input("Enter keywords (comma-separated): ")
user_keywords_list = user_keywords.split(',')

# Calculate cosine similarity and score the OCR results for all PDFs (multiply by 100)
vectorizer_all_pdfs = CountVectorizer().fit_transform([result["text"] for result in pdf_ocr_results.values()] + user_keywords_list)
cosine_similarities_all_pdfs = cosine_similarity(vectorizer_all_pdfs)

# Score the OCR results based on cosine similarity for all PDFs
for i, (pdf_file, result) in enumerate(pdf_ocr_results.items()):
    similarity_score_all_pdfs = cosine_similarities_all_pdfs[i][-len(user_keywords_list):]
    result["score"] = round(sum(similarity_score_all_pdfs) * 100, 1)

    # Identify matched keywords for the current PDF
    matched_keywords = [user_keywords_list[j] for j, score in enumerate(similarity_score_all_pdfs) if score > 0]
    result["matched_keywords"] = matched_keywords

# Sort OCR results by the score in descending order for all PDFs
sorted_ocr_results_all_pdfs = sorted(pdf_ocr_results.items(), key=lambda x: x[1]["score"], reverse=True)

# Display the sorted JSON data for all PDFs
print("\nSorted PDFs:")
for pdf_file, result in sorted_ocr_results_all_pdfs:
    print(f"PDF: {pdf_file} | Score: {result['score']} | Image Path: {result['img_path']} | Matched Keywords: {result['matched_keywords']}")

    # Generate a summary for the OCR text
    parser = PlaintextParser.from_string(result['text'], Tokenizer("english"))
    summarizer = LuhnSummarizer()

    # Set the desired number of words in the summary (adjust the value as needed)
    summary_word_count = 300  # For example, 300 words in the summary
    summary = summarizer(parser.document, sentences_count=summary_word_count)

    print("Summary:", " ".join(str(sentence) for sentence in summary))
    print("\n" + "=" * 50 + "\n")  # Separator
