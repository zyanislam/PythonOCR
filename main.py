import json
import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pytesseract.pytesseract.tesseract_cmd = text = r'/usr/bin/tesseract'

pdfs_directory = "/home/quantum/Downloads/Projects/PythonOCR/pdfs"
output_folder = "images"

# Create the "images" folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all PDF files in the "pdfs" directory
pdf_files = [f for f in os.listdir(pdfs_directory) if f.endswith('.pdf')]

# Create a list to store the OCR results and images for all PDFs
all_ocr_results = []

# Loop through each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdfs_directory, pdf_file)

    # Convert PDF to images
    images = convert_from_path(pdf_path)

    result_dict = {}
    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(img_rgb)
        result_dict[f"page_{i+1}"] = text

        # Save the image to the "images" folder
        img.save(os.path.join(output_folder, f"{pdf_file}_page_{i+1}.png"))

    # Create a list to store the OCR results and images for the current PDF
    ocr_results = []
    for key, value in result_dict.items():
        img_path = os.path.join(output_folder, f"{pdf_file}_page_{key}.png")
        ocr_results.append({"page": key, "text": value, "img_path": img_path, "pdf_file": pdf_file})

    # Extend the list of OCR results for all PDFs
    all_ocr_results.extend(ocr_results)

# Convert the list to JSON for all PDFs
json_data_all_pdfs = json.dumps(all_ocr_results, indent=2)

# Print or save the JSON data for all PDFs
print("JSON Data for All PDFs:")
print(json_data_all_pdfs)

# Prompt the user for keywords
user_keywords = input("Enter keywords (comma-separated): ")
user_keywords_list = user_keywords.split(',')

# Calculate cosine similarity and score the OCR results for all PDFs
vectorizer_all_pdfs = CountVectorizer().fit_transform([result["text"] for result in all_ocr_results] + user_keywords_list)
cosine_similarities_all_pdfs = cosine_similarity(vectorizer_all_pdfs)

# Score the OCR results based on cosine similarity for all PDFs
for i, result in enumerate(all_ocr_results):
    similarity_score_all_pdfs = cosine_similarities_all_pdfs[i][-len(user_keywords_list):]
    result["score"] = sum(similarity_score_all_pdfs)

# Sort OCR results by the score in descending order for all PDFs
sorted_ocr_results_all_pdfs = sorted(all_ocr_results, key=lambda x: x["score"], reverse=True)

# Display only the names of the pages in the sorted JSON data for all PDFs
print("\nSorted Pages for All PDFs:")
for result in sorted_ocr_results_all_pdfs:
    print(f"Page: {result['page']} | Score: {result['score']} | PDF: {result['pdf_file']}")
