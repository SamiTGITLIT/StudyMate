import json
import re
import time
import requests
from docx import Document
import os

API_KEY = "-"
MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DOCX = os.path.join(SCRIPT_DIR, "mcqs_input.docx")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "mcq_tagged_output.json")
BATCH_SIZE = 10
DELAY = 5

BIO_UNITS = {
    "Grade 9": [
        "Biology and Technology", "Cell Biology", "Human Biology and Health",
        "Micro-organisms and Disease", "Classification", "Environment"
    ],
    "Grade 10": [
        "Biotechnology", "Heredity", "Human Biology and Health",
        "Food Making and Growth in Plants", "Conservation of Natural Resources"
    ],
    "Grade 11": [
        "Biology and Technology", "Characteristics of Animals", "Enzymes",
        "Genetics", "The Human Body Systems", "Population and Natural Resources"
    ],
    "Grade 12": [
        "Microorganisms", "Ecology", "Genetics", "Evolution", "Behaviour",
        "Climate Change"
    ]
}
unit_list_str = "\n".join(f"{g}: " + ", ".join(u) for g, u in BIO_UNITS.items())

SYSTEM_PROMPT = f"""
You are an educational AI assistant. You have access to these Biology units:
{unit_list_str}

For each MCQ, return a JSON object:
- question: the full question string
- options: list of all 4 options
- units: all relevant units (list all Grade 9-12 units that apply)
- topic: the specific topic or portion within that unit
- bloom_level: Bloom's Taxonomy level (Remembering, Understanding, Applying, etc.)

Respond ONLY with a JSON list:
[
  {{
    "question": "...",
    "options": ["A...", "B...", "C...", "D..."],
    "units": ["..."],
    "topic": "...",
    "bloom_level": "..."
  }},
  ...
]
"""

def parse_docx(path):
    text = "\n".join(p.text.strip() for p in Document(path).paragraphs if p.text.strip())
    mcqs = []
    matches = re.split(r"(?=\d+\s*\.\s)", text)
    for match in matches:
        lines = match.strip().split("\n")
        if len(lines) >= 5:
            q_line = lines[0]
            opt_lines = lines[1:5]
            question = re.sub(r"^\d+\s*\.\s*", "", q_line)
            options = [re.sub(r"^[A-Da-d][\s\.\)\-]*", "", o.strip()) for o in opt_lines]
            if len(options) == 4:
                mcqs.append({"question": question, "options": options})
    print(f"Parsed {len(mcqs)} MCQs from {os.path.basename(path)}")
    return mcqs

def call_openrouter(batch):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    content = "\n".join(f"{i+1}. {q['question']}\nOptions: {', '.join(q['options'])}" for i, q in enumerate(batch))
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API error: {e}")
        return None

def parse_safe_json(content):
    objects = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
    results = []
    for block in objects:
        try:
            results.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    return results

def process_mcqs(path):
    mcqs = parse_docx(path)
    tagged = []
    total = len(mcqs)
    print(f"Processing {total} MCQs in {(total-1)//BATCH_SIZE+1} batches of {BATCH_SIZE}…")
    for i in range(0, total, BATCH_SIZE):
        batch = mcqs[i:i+BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE+1}/{(total-1)//BATCH_SIZE+1}…")
        response = call_openrouter(batch)
        if response:
            parsed = parse_safe_json(response)
            print(f" ✓ Batch {i//BATCH_SIZE+1} done.")
            tagged.extend(parsed)
        else:
            print(f" ✗ No response for batch {i//BATCH_SIZE+1}.")
        time.sleep(DELAY)
    return tagged

if __name__ == "__main__":
    results = process_mcqs(INPUT_DOCX)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(results)} tagged MCQs to {OUTPUT_JSON}")
