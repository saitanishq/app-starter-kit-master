import streamlit as st
import boto3
import json
import openai
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize AWS S3 client and SentenceTransformer model
s3_client = boto3.client('s3')
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_json_from_s3(bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))

def write_json_to_s3(data, bucket, key):
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode('utf-8'))

def normalize_text(text):
    return text.lower().strip()

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

def search_question_in_bucket(question, bucket, prefix, keywords):
    # Check if the question contains any relevant keywords
    if not any(keyword in question.lower() for keyword in keywords):
        return False, "This question does not pertain to female health."

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
        return False, f"GPT-generated answer: {generated_answer}"

# Define keywords related to female health
female_health_keywords = ["female", "woman", "women", "girl", "ovary", "ovaries", "menstruation", "pregnancy", "breast"]

st.title('Health AI Davinci')
bucket_name = st.secrets["BUCKET_NAME"]
prefix = st.secrets["PREFIX"]

if "query_history" not in st.session_state:
    st.session_state.query_history = []

def add_query_to_history(question):
    st.session_state.query_history.append(question)

with st.sidebar:
    st.write("## Question History")
    for query in st.session_state.query_history:
        st.write(query)

question = st.text_input("Ask a question:", "")
if question:
    add_query_to_history(question)
    with st.spinner('Searching for your answer...'):
        found, answer = search_question_in_bucket(question, bucket_name, prefix, female_health_keywords)
        if found:
            st.success("Found answer in the database:")
        else:
            st.success("Answer generated by GPT:")
        st.write(answer)
