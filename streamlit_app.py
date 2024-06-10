import streamlit as st
import boto3
import json
import openai
import regex as re
import string
import speech_recognition as sr
from datetime import datetime
from pytrends.request import TrendReq
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
import nltk
from nltk.tokenize import sent_tokenize
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')
import numpy as np

# Initialize AWS S3 client and SentenceTransformer model
s3_client = boto3.client('s3')
model = SentenceTransformer('all-MiniLM-L6-v2')
pytrends = TrendReq(hl='en-US', tz=360)

def read_json_from_s3(bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))

def write_json_to_s3(data, bucket, key):
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode('utf-8'))

def get_next_filename(bucket, prefix):
    """
    Generates the next filename in sequence based on the highest number found in existing filenames within an S3 prefix.
    Assumes filenames are in the format 'fXXX-20231015_qa.json' where XXX is a number.
    """
    max_number = 0
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            filename = obj['Key'].split('/')[-1]  # Get the filename part of the key
            parts = filename.split('-')
            number_part = parts[0][1:]  # Remove the 'f' prefix
            if number_part.isdigit():
                number = int(number_part)
                if number > max_number:
                    max_number = number

    next_number = max_number + 1
    next_filename = f"f{next_number:03}-20231015_qa.json"
    return next_filename

def normalize_text(text):
    """
    Converts text to lower case, removes punctuation, and strips leading and trailing spaces.
    Handles None by returning an empty string.
    """
    if text is None:
        return ""
    remove_punct = str.maketrans('', '', string.punctuation)
    return text.lower().translate(remove_punct).strip()


def get_most_similar_question(input_question, questions):
    question_texts = list(questions.keys())
    if not hasattr(get_most_similar_question, 'embeddings'):
        get_most_similar_question.embeddings = model.encode(question_texts, convert_to_tensor=True)
    question_embedding = model.encode(input_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, get_most_similar_question.embeddings)[0]
    similarities = similarities.cpu().numpy()
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]
    if similarity_score >= 0.5:
        most_similar_question = question_texts[most_similar_index]
        return most_similar_question, questions[most_similar_question], similarity_score.item()
    return None, None, 0

def generate_answer_with_gpt(question, context):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

def search_question_in_bucket(question, bucket, prefix):

    questions = {}
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            data = read_json_from_s3(bucket, obj['Key'])
            normalized_question = normalize_text(data.get('q'))
            questions[normalized_question] = data.get('a')
    most_similar_question, answer, score = get_most_similar_question(normalize_text(question), questions)
    if most_similar_question:
        return True, answer
    else:
        context = "Provide information relevant to user queries based on previous knowledge."
        generated_answer = generate_answer_with_gpt(question, context)
        # Generate the next filename
        next_filename = get_next_filename(bucket, prefix)
        qa_data = {'q': question, 'a': generated_answer}
        write_json_to_s3(qa_data, bucket, f"{prefix}{next_filename}")
        return False, f"GPT-generated answer: {generated_answer}"

col1, col2 = st.columns([1, 3])

# Display the logo in the first column
logo_path = 'Davincilogo.jpeg'  # Change to the location of your logo file
col1.image(logo_path, width=150)  # Adjust the width to suit your layout

# Display the title in the second column
col2.title('Health AI Davinci')
bucket_name = st.secrets["BUCKET_NAME"]
prefix = st.secrets["PREFIX"]

# Initialize a zero-shot classification model
try:
    classifier = hf_pipeline("zero-shot-classification")
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")

if "query_history" not in st.session_state:
    st.session_state.query_history = []

def add_query_to_history(question):
    st.session_state.query_history.append(question)

def is_womens_health_question(question, topic):
    if classifier:
        result = classifier(question, candidate_labels=[topic, "other"])
        return result['labels'][0] == topic and result['scores'][0] > 0.75
    return False

with st.sidebar:
    st.write("## Question History")
    for query in st.session_state.query_history:
        st.write(query)


question = st.text_input("Ask a question about women's health:", "")

# Suggestive questions as clickable options
suggestive_questions = [
    "What is menstruation and how long does it last?",
    "What are common signs of PCOS?",
    "What are the signs of endometriosis?",
    "What is the follicular phase and how long does it last?",
    "What is ovulation and how long does it last?",
    "What is the luteal phase and how long does it last?",
    "How does weight gain impact the menstrual cycle?",
    "How does obesity impact the menstrual cycle?",
    "Diet recommendations during menstruation",
    "Fitness and workout recommendations during menstruation"
]

st.write("## Explore Topics")


for q in suggestive_questions:
    if st.button(q):
        question = q  # Update the question variable with the button's question
        add_query_to_history(question)  # Add the question to history
        break  # Break after a button click to prevent multiple handlers from running
def clean_and_parse_answer(answer):
    # Strip whitespaces first to clean up the string
    answer = answer.strip()
    
    # Remove 'a:"' from the beginning
    answer = re.sub(r'^[^a-zA-Z]*a:"', '', answer)
    
    # Remove '}"' from the end
    answer = re.sub(r'"}\s*$', '', answer)
    
    return answer

def format_key_points(text):
    # Split main subtopics using double newlines
    subtopics = re.split(r'\n\n', text)
    formatted_text = "<div>"

    for subtopic in subtopics:
        # Identify main subtopics and further split by newline to separate subpoints
        lines = subtopic.strip().split("\n")
        if lines:
            # Use the first line as the main title, bold it
            formatted_text += f"<h6 style='font-weight: bold;'>{lines[0].strip()}</h4><ul>"
            # Process each subsequent line as a separate point
            for line in lines[1:]:
                # Handle subpoints that may have additional descriptions separated by colons
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    formatted_text += f"<li><strong>{parts[0].strip()}:</strong> {parts[1].strip()}</li>"
                else:
                    formatted_text += f"<li>{parts[0].strip()}</li>"
            formatted_text += "</ul>"

    formatted_text += "</div>"
    return formatted_text

def display_answer_with_html(raw_answer):
    # Clean and parse the answer
    answer = clean_and_parse_answer(raw_answer)
    
    # Tokenize the answer into sentences
    sentences = sent_tokenize(answer)
    intro = sentences[0]  # Assuming the first sentence is the introduction
    
    # Heuristic: Assume longer sentences contain more information
    key_points = [sentence for sentence in sentences[1:] if len(sentence.split()) > 10]
    
    # Limit to top 3 key points for simplicity
    key_points = key_points[:3]

    # Format the key points as HTML list items
    key_points_html = ''.join(f'<li>{point}</li>' for point in key_points)

    # Detailed explanation is the text following the intro and key points, handled differently if needed
    detailed_explanation = format_key_points(' '.join(sentences[1:]))

    formatted_answer = f"""
    <div style="font-family: Arial; font-size: 16px; color: #444;">
        <h3>Answer Summary</h3>
        <p><strong>{intro}</strong></p>
        <h4>Key Points:</h4>
        <ul>{key_points_html}</ul>
        <h3>More detailed explanation:</h3>
        <div style="margin-left: 20px;">{detailed_explanation}</div>
    </div>
    """
    st.markdown(formatted_answer, unsafe_allow_html=True)

# Process the question either typed or clicked
if question:
    with st.spinner('Searching for your answer...'):
        if is_womens_health_question(question, "women's health"):
            found, answer = search_question_in_bucket(question, bucket_name, prefix)
            if found:
                st.success("Found answer in the database:")
                display_answer_with_html(answer)
            else:
                st.success("Answer generated by GPT:")
                display_answer_with_html(answer)
        else:
            st.error("Your question does not appear to relate to women's health.")
