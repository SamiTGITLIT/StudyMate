import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ← EDIT: folder where your tagged-MCQ JSON files live
JSON_FOLDER  = './TaggedOutput'
# ← EDIT: where ChromaDB should persist its on-disk store
CHROMA_DIR   = './chromadb'
# ← EDIT: your LM Studio local server URL and Mistral model ID
LMSTUDIO_API = "http://192.168.1.14:1234/v1"
LM_MODEL     = "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF"

# Hide the Streamlit developer toolbar (including “Deploy” button) 
st.set_page_config(toolbar_mode="never")  # must be first Streamlit call :contentReference[oaicite:0]{index=0}

# 1) Sentence-Transformers embedder
class CustomEmbeddingFunction(EmbeddingFunction[list[str]]):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

embedding_function = CustomEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")

# 2) Initialize ChromaDB persistent client
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(
    name="mcq_collection",
    embedding_function=embedding_function
)

# 3) Embed & cache MCQs on first run
existing_ids = set(collection.get(include=[]).get("ids", []))
for fn in os.listdir(JSON_FOLDER):
    if not fn.endswith(".json"):
        continue
    data = json.load(open(os.path.join(JSON_FOLDER, fn), encoding="utf-8"))
    for i, mcq in enumerate(data):
        qid = mcq.get("id", f"{fn}_{i}")
        if qid in existing_ids:
            continue
        collection.add(
            documents=[mcq["question"]],
            metadatas=[{
                "unit":        mcq.get("unit", ""),
                "grade":       mcq.get("grade", ""),
                "bloom_level": mcq.get("bloom_level", "")
            }],
            ids=[qid]
        )

# 4) OpenAI-compatible LM Studio client with raised timeout
llm = OpenAI(base_url=LMSTUDIO_API, api_key="lm-studio", timeout=120.0)

# 5) Retry-wrapped LLM invocation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def safe_query_llm(messages, stream=False):
    return llm.chat.completions.create(
        model=LM_MODEL,
        messages=messages,
        temperature=0.2,
        stream=stream
    )

def query_llm(messages, stream=False):
    resp = safe_query_llm(messages, stream=stream)
    if stream:
        output = ""
        for chunk in resp:
            delta = chunk.choices[0].delta.get("content", "")
            st.write(delta)
            output += delta
        return output
    else:
        return resp.choices[0].message.content

# 6) RAG retrieval helper with $and filter
def retrieve_mcqs(unit, grade, bloom, k=5):
    emb = embedding_function([f"{unit} {grade} {bloom}"])[0]
    where_filter = {"$and": [
        {"unit": unit},
        {"grade": grade},
        {"bloom_level": bloom}
    ]}
    res = collection.query(
        query_embeddings=[emb],
        n_results=k,
        where=where_filter
    )
    return res["documents"][0]

# 7) Prompt templates
def make_short_note_prompt(mcqs, unit, grade, bloom):
    ctx = "\n".join(f"- {q}" for q in mcqs)
    return [
        {"role":"system", "content":"You are an expert high-school tutor."},
        {"role":"user",   "content":
         f"Generate concise study notes on {unit} for {grade} at the {bloom} level.\nContext:\n{ctx}"}
    ]

def make_quiz_prompt(mcqs, unit, grade, bloom):
    ctx = "\n".join(f"- {q}" for q in mcqs)
    return [
        {"role":"system", "content":"You are an exam-question generator."},
        {"role":"user",   "content":
         f"Create one multiple-choice question on {unit} for {grade} at the {bloom} level.\nContext:\n{ctx}"}
    ]

# 8) Streamlit GUI with dynamic Unit dropdown
st.title("AI StudyMate")

mode  = st.selectbox("Mode", ["Short Note", "Quiz"])
grade = st.selectbox("Grade", ["Grade 9", "Grade 10", "Grade 11", "Grade 12"])

# Dynamically fetch valid units so user must pick one exactly as stored :contentReference[oaicite:1]{index=1}
all_metas = collection.get(include=["metadatas"])["metadatas"]
units = sorted({m["unit"] for m in all_metas})
unit  = st.selectbox("Unit", units, index=None, label_visibility="visible")  # index=None → no default :contentReference[oaicite:2]{index=2}

bloom = st.selectbox("Bloom Level", ["Understanding", "Application", "Analysis"])

if st.button("Generate"):
    if not unit:
        st.error("Please select a Unit from the dropdown before generating.")  # user feedback :contentReference[oaicite:3]{index=3}
    else:
        mcqs = retrieve_mcqs(unit, grade, bloom, k=5)
        msgs = make_short_note_prompt(mcqs, unit, grade, bloom) if mode=="Short Note" else make_quiz_prompt(mcqs, unit, grade, bloom)
        result = query_llm(msgs, stream=False)
        st.markdown(result)
