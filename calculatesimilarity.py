# pip install sentence-transformers
# pip install pyspellchecker
# pip install nltk sentence-transformers torch rapidfuzz pyspellchecker
# pip install PyPDF2
# pip install nltk sentence-transformers torch rapidfuzz pyspellchecker

import nltk
from nltk.corpus import wordnet as wn  # For synonyms via WordNet
from nltk.corpus import stopwords  # Stopwords, if needed
from spellchecker import SpellChecker  # For typo detection and correction
from nltk import pos_tag
# Sentence-BERT and PyTorch for sentence embeddings
import torch
from sentence_transformers import SentenceTransformer, util

#Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('punkt_tab')            # For tokenization
# nltk.download('wordnet')          # For WordNet
# nltk.download('omw-1.4')          # For multilingual WordNet
# nltk.download('averaged_perceptron_tagger')  # For part-of-speech tagging
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')
print("Start")
# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define a function to get synonyms of a word
# pip install requests
import requests

def fetch_synonyms_from_thesaurus(word):
    url = f"https://api.datamuse.com/sug?s={word}"
    response = requests.get(url)
    data = response.json()
    synonyms = [item['word'] for item in data if 'word' in item]
    return synonyms
def get_synonyms(phrase, threshold=0.7):
    synonyms = {}
    words = list(phrase.split(" "))

    for word in words:
        synonyms[word] = set()
        synonyms[word].add(word)

        # Fetch synonyms from the thesaurus API
        thesaurus_synonyms = fetch_synonyms_from_thesaurus(word)
        synonyms[word].update(thesaurus_synonyms)  # Add fetched synonyms

        # Existing WordNet logic
        pos = pos_tag([word])[0][1][0].lower()
        pos_map = {'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV}

        word_synsets = wn.synsets(word, pos=pos_map.get(pos, None))
        for syn in word_synsets:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms[word].add(lemma.name())

    # Convert sets to lists
    for key in synonyms:
        synonyms[key] = list(synonyms[key])
    return synonyms





from difflib import get_close_matches

# Function to correct typos by comparing words against a list of correct names
def check_typos(word, correct_names):
    spell = SpellChecker()
    spell.word_frequency.load_words(correct_names)

    # Use difflib to find the closest match from correct_names
    close_matches = get_close_matches(word, correct_names, n=1, cutoff=0.6)
    if close_matches:
        return close_matches[0]

    # Fallback to the spell checker if no close match is found
    return spell.correction(word)



def has_required_keywords(sentence, main_key):
    sentence = sentence.lower()  # Lowercase the sentence
    main_keyword = get_synonyms(main_key)  # Synonyms based on the lowercase key
    for synonym in main_keyword:
        if synonym in sentence:
            return True
    return False

def get_sentence_embedding(sentence):
    # Get embeddings for the sentence using Sentence-BERT
    return model.encode(sentence, convert_to_tensor=True)




##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################




def calculate_similarity(candidate, reference,names, sub_keys, main_keys):
    # Get embeddings for both sentences
    candidate_embedding = get_sentence_embedding(candidate)
    reference_embedding = get_sentence_embedding(reference)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(candidate_embedding, reference_embedding).item()
    
    
    len_candidate = len(candidate)
    len_reference = len(reference)
    threshold= 0.70 * len_reference
    # If candidate length is less than 70% of reference length, reduce marks
    if len_candidate < threshold:
        length_ratio = len_candidate / len_reference
        penalty = (0.70 - length_ratio)*1.5   # Apply a penalty based on how much smaller
        similarity -= penalty
###############################################################################################################################################################################
    if similarity>0.7:
      similarity=1.0
    elif similarity < 0.45:
        return 0
    #Check for names or stuff that won't have a synonym(Mainly names)
    if names:
      for name in names:
          name_typo = check_typos(candidate, name)
          if not name_typo:
              similarity -= 0.01 #check if name or any typos of name exists in the candidate,if so, then depending on the threshold, give marks
###############################################################################################################################################################################
    # **Partial Credit for Sub Keywords**
    # Check sub keywords and count how many are present in the candidate
    missing_sub_keywords = 0
    # Sub-keywords Check
    # Sub-keywords Check
    if sub_keys:
        for sub_key in sub_keys:
            sub_key_lower = sub_key.lower()  # Convert sub_key to lowercase
            sub_keyword_synonyms = get_synonyms(sub_key_lower)  # Get synonyms as a list or dictionary

            # Check if any synonym of sub_key is in the candidate
            found = False
            for synonym in sub_keyword_synonyms[sub_key_lower]:  # Loop through all synonyms of the sub_key
                if synonym in candidate:
                    found = True
                    break  # Stop checking further if a synonym is found

            if not found:  # If no synonym was found, increment the missing sub_keywords counter
                missing_sub_keywords += 1


        # Adjust similarity based on how many sub keywords are correct
        if missing_sub_keywords > 0:
              total_deduction = missing_sub_keywords * (0.05)
              similarity -= total_deduction

###############################################################################################################################################################################
    # Check for main keyword (or its synonyms) in the candidate
    count=0
    if main_keys:
      for main_key in main_keys:
          main_key_lower = main_key.lower()  # Convert to lowercase first
          main_keyword_synonyms = get_synonyms(main_key_lower)  # Get synonyms as a list or dictionary

          # Check if any synonym of main_key is in the candidate
          found = False
          if main_key_lower in main_keyword_synonyms:
            for synonym in main_keyword_synonyms[main_key_lower]:  # Loop through all synonyms
                if synonym in candidate:
                    found = True
                    break  # Stop checking further if a synonym is found

          if not found:  # If no synonym was found, increment the counter
              count += 1


    if count>0:
        # print(4)
        if len(main_keys)==1:
            print(similarity)
            if similarity>=0.5:
                return 0.5
            else:
                return 0.0
        similarity -= count * 0.25


##############################################################################################################################################################################

    similarity = max(similarity, 0)
    if similarity>0.75:
      return 1.0
    print("Final score incoming!")
    return round(similarity, 2)  # Return the final similarity score


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# Function to replace shortforms with full forms in text
def replace_shortforms(text, shortforms):
    words = text.split()
    # Replace each word if it's a key in shortforms dictionary
    replaced_words = [shortforms[word] if word in shortforms else word for word in words]
    return ' '.join(replaced_words)

# If user wants to upload a file/image instead of text
#pip install easyocr pillow PyPDF2 python-docx

# Step 2: Import necessary libraries
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import easyocr  # Using EasyOCR as an alternative OCR option
import os

# Initialize EasyOCR reader (for image text extraction)
reader = easyocr.Reader(['en'])  # Specify the language, e.g., 'en' for English

# Function to extract text from a .pdf file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a .docx file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from a .txt file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Function to extract text from an image file using EasyOCR
def extract_text_from_image(image_path):
    try:
        result = reader.readtext(image_path)

        # Group words into lines based on vertical proximity
        lines = []
        for bbox, text, _ in result:
            x_min, y_min = bbox[0]  # Get top-left coordinates
            added = False

            # Try to append text to an existing line if it's close in the Y-axis
            for line in lines:
                if abs(line["y"] - y_min) < 15:  # Adjust threshold for grouping
                    line["text"].append((x_min, text))
                    added = True
                    break
            
            # If no matching line found, create a new one
            if not added:
                lines.append({"y": y_min, "text": [(x_min, text)]})

        # Sort lines top-to-bottom, and words left-to-right within each line
        lines.sort(key=lambda x: x["y"])
        final_text = "\n".join(" ".join([word[1] for word in sorted(line["text"])]) for line in lines)

        return final_text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


# Function to split text into separate answers based on the "Answer" keyword
def separate_answers(text):
    # Split the text on the word "Answer" with proper reattachment of the keyword
    split_answers = text.split("Answer")
    answers = []

    for part in split_answers:
        cleaned_part = part.strip()
        if cleaned_part:  # Only add non-empty parts
            answers.append(f"{cleaned_part}")

    return answers

# Main function to determine file type, extract text, and separate answers
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        text = extract_text_from_txt(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        text = extract_text_from_image(file_path)  # Use EasyOCR
    else:
        return "Unsupported file type. Please provide a .pdf, .docx, .txt, .png, .jpg, or .jpeg file."

    # Separate answers if possible
    return separate_answers(text)


#######################################################################################################################################
# print("Finished")
# candidate_answer="Hello I am Shruti"
# reference_answer="Hello I'm Shruti"
# names=['Shruti']
# sub_keys=[]
# main_keys=['Hello']

# score = calculate_similarity(
#         candidate=candidate_answer.lower(),
#         reference=reference_answer.lower(),
#         names=names,
#         sub_keys=sub_keys,
#         main_keys=main_keys
#     )
# print(score)