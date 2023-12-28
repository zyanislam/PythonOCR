import json
import os
import cv2
import pytesseract
import numpy as np

from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pytesseract.pytesseract.tesseract_cmd = text = r'/usr/bin/tesseract'

pdf_path = "/home/quantum/Downloads/Projects/PythonOCR/pdfs/Resume - Rezab Ud Dawla.pdf"
images = convert_from_path(pdf_path)

output_folder = "images"

result_dict = {}
for i, img in enumerate(images):
    img_rgb = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb)
    result_dict[f"page_{i+1}"] = text

    # Save the image to the "images" folder
    img.save(os.path.join(output_folder, f"page_{i+1}.png"))

image1 = cv2.imread("images/image_1.png")
image2 = cv2.imread("images/image_2.png")
image3 = cv2.imread("images/page_1.png")
image4 = cv2.imread("images/page_2.png")

img_1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
img_2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
img_3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
img_4_rgb = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

print(pytesseract.image_to_string(img_1_rgb))
print(pytesseract.image_to_string(img_2_rgb))
print(pytesseract.image_to_string(img_3_rgb))
print(pytesseract.image_to_string(img_4_rgb))

text_1 = pytesseract.image_to_string(img_1_rgb)
text_2 = pytesseract.image_to_string(img_2_rgb)
text_3 = pytesseract.image_to_string(img_3_rgb)
text_4 = pytesseract.image_to_string(img_4_rgb)

# Create a dictionary to store the OCR results
result_dict = {
    "image_1": text_1,
    "image_2": text_2,
    "image_3": text_3,
    "image_4": text_4
}

# Convert the dictionary to JSON
json_data = json.dumps(result_dict, indent=2)

# Print or save the JSON data
print("JSON Data:")
print(json_data)

# Prompt the user for keywords
user_keywords = input("Enter keywords (comma-separated): ")
user_keywords_list = user_keywords.split(',')

# Calculate cosine similarity and filter the JSON data
filtered_data = {}
vectorizer = CountVectorizer().fit_transform([text_1, text_2, text_3, text_4] + user_keywords_list)
cosine_similarities = cosine_similarity(vectorizer)

for i, key in enumerate(result_dict.keys()):
    similarity_score = cosine_similarities[i][-len(user_keywords_list):]
    if any(score > 0 for score in similarity_score):
        filtered_data[key] = result_dict[key]

filtered_json_data = json.dumps(filtered_data, indent=2)

print("\nFiltered JSON Data:")
print(filtered_json_data)

# with open("output.json", "w") as json_file:
#     json_file.write(json_data)

# result = pytesseract.image_to_boxes(img_1_rgb)
# ih, iw, ic = image1.shape
# for box in result.splitlines():
#     box = box.split(' ')
#     print(box)
#
#     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#     cv2.rectangle(image1, (x, ih-y), (w, ih-h), (0, 255, 0), 2)

# cv2.imshow("Input", image1)
# cv2.waitKey(0)
#
# cv2.imshow("Input", image2)
# cv2.waitKey(0)