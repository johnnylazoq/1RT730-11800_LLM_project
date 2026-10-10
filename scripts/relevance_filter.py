import json
from sentence_transformers import SentenceTransformer, util

claim = "Does vitamin C prevent the common cold?"

# Load retrieved papers
with open("retrieved_papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

paper_texts = []

# title or abstract to be embedded for relevance scoring
for paper in papers:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    paper_texts.append(f"{title}. {abstract}")

claim_embedding = model.encode(
    claim,
    convert_to_tensor=True
)

paper_embeddings = model.encode(
    paper_texts,
    convert_to_tensor=True
)

similarities = util.cos_sim(
    claim_embedding,
    paper_embeddings
)[0]

for paper, score in zip(papers, similarities):
    paper["relevance_score"] = float(score)

papers.sort(
    key=lambda x: x["relevance_score"],
    reverse=True
)

top_papers = []

for paper in papers:
    abstract = paper.get("abstract")
    open_access_pdf = paper.get("open_access_pdf")

    if abstract or open_access_pdf:
        if len(top_papers) < 5:
            top_papers.append(paper)
   
            

# Save top 5 that are high relevance + abstract OR open-access full text
with open("top_5_papers.json", "w", encoding="utf-8") as f:
    json.dump(top_papers, f, indent=2, ensure_ascii=False)





