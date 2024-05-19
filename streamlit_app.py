import streamlit as st
import boto3
import json
import openai
import torch
import string
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
import numpy as np

@st.cache
def load_model():
	  return torch.load("model.pt")

model = load_model()

# Initialize AWS S3 client and SentenceTransformer model
s3_client = boto3.client('s3')
model = SentenceTransformer('all-MiniLM-L6-v2')

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
        qa_data = {'question': question, 'answer': generated_answer}
        write_json_to_s3(qa_data, bucket, f"{prefix}{next_filename}")
        return False, f"GPT-generated answer: {generated_answer}"

st.title('Health AI Davinci')
bucket_name = st.secrets["BUCKET_NAME"]
prefix = st.secrets["PREFIX"]

# Initialize a zero-shot classification model
try:
    model_name = "facebook/bart-large-mnli"
    revision = "c626438"
    classifier = hf_pipeline("zero-shot-classification", model=model_name, revision=revision)
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

question = st.text_input("Ask a question:", "")
if question:
    add_query_to_history(question)
    with st.spinner('Searching for your answer...'):
        if is_womens_health_question(question, "women's health"):
            found, answer = search_question_in_bucket(question, bucket_name, prefix)
            if found:
                st.success("Found answer in the database:")
                st.write(answer)
            else:
                st.success("Answer generated by GPT:")
                st.write(answer)
        else:
            st.error("Your question does not appear to relate to women's health.")
