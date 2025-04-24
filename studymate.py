#!/usr/bin/env python3
import os, json, argparse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ← adjust paths/URLs as needed
JSON_FOLDER  = './TaggedOutput'
CHROMA_DIR   = './chromadb'
LMSTUDIO_API = "http://192.168.1.14:1234"
LM_MODEL     = "mistral-7b-instruct-v0.3"

# 1) EmbeddingFunction for ChromaDB
class CustomEmbeddingFunction(EmbeddingFunction[list[str]]):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

embedding_function = CustomEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")

# 2) Initialize ChromaDB and ingest MCQs
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="mcq_collection", embedding_function=embedding_function)

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
                "unit":        mcq.get("unit",""),
                "grade":       mcq.get("grade",""),
                "bloom_level": mcq.get("bloom_level","")
            }],
            ids=[qid]
        )

# 3) Fetch valid unit names from metadata (must do this before argparse)
all_metas = collection.get(include=["metadatas"])["metadatas"]
units = sorted({m["unit"] for m in all_metas})   # ← dynamic list of real units :contentReference[oaicite:5]{index=5}

# 4) Build CLI with argparse, using the real units as choices
parser = argparse.ArgumentParser(prog="StudyMate CLI", description="Generate study notes or quiz questions via RAG+LLM")  # :contentReference[oaicite:6]{index=6}
parser.add_argument(
    "--unit",
    required=True,
    choices=[
        "Genetics",
        "Ecology",
        "Cell Biology",
        "Evolution",
        "Photosynthesis"
    ],
    help="Unit name (must match one of the metadata values exactly)"                 # 
)
parser.add_argument(
    "--grade",
    required=True,
    choices=["Grade 9","Grade 10","Grade 11","Grade 12"],
    help="Grade level"
)
parser.add_argument(
    "--bloom",
    required=True,
    choices=["Understanding","Application","Analysis"],
    help="Bloom taxonomy level"
)
parser.add_argument(
    "--mode",
    default="Short Note",
    choices=["Short Note","Quiz"],
    help="Output mode"
)
args = parser.parse_args()  # 

# 5) OpenAI-compatible client with extended timeout
llm = OpenAI(base_url=LMSTUDIO_API, api_key="lm-studio", timeout=120.0)               # 

# 6) Retry logic for LLM calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))     # 
def safe_query_llm(messages, stream=False):
    return llm.chat.completions.create(model=LM_MODEL, messages=messages, temperature=0.2, stream=stream)

def query_llm(messages, stream=False):
    resp = safe_query_llm(messages, stream=stream)
    if stream:
        out = ""
        for chunk in resp:
            text = chunk.choices[0].delta.get("content","")
            print(text, end="", flush=True)
            out += text
        print()
        return out
    return resp.choices[0].message.content

# 7) RAG retrieval helper with $and filter
def retrieve_mcqs(unit, grade, bloom, k=5):
    emb = embedding_function([f"{unit} {grade} {bloom}"])[0]
    where_filter = {"$and":[{"unit": unit}, {"grade": grade}, {"bloom_level": bloom}]}     # :contentReference[oaicite:7]{index=7}
    res = collection.query(query_embeddings=[emb], n_results=k, where=where_filter)
    return res["documents"][0]

# 8) Prompt templates
def make_short_note_prompt(mcqs, unit, grade, bloom):
    ctx = "\n".join(f"- {q}" for q in mcqs)
    return [
      {"role":"system","content":"You are an expert high-school tutor."},
      {"role":"user","content":f"Generate concise study notes on {unit} for {grade} at the {bloom} level.\nContext:\n{ctx}"}
    ]

def make_quiz_prompt(mcqs, unit, grade, bloom):
    ctx = "\n".join(f"- {q}" for q in mcqs)
    return [
      {"role":"system","content":"You are an exam-question generator."},
      {"role":"user","content":f"Create one multiple-choice question on {unit} for {grade} at the {bloom} level.\nContext:\n{ctx}"}
    ]

# 9) Execute retrieval + LLM and print
mcqs = retrieve_mcqs(args.unit, args.grade, args.bloom, k=5)
msgs = make_short_note_prompt(mcqs, args.unit, args.grade, args.bloom) if args.mode=="Short Note" else make_quiz_prompt(mcqs, args.unit, args.grade, args.bloom)
output = query_llm(msgs, stream=False)
print("\n" + output)
