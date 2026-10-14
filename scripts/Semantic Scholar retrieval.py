import os
import json
import requests

query = "vitamin C prevention common cold"

url = "https://api.semanticscholar.org/graph/v1/paper/search"

headers = {
    "x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]
}

params = {
    "query": query,
    "limit": 20,
    "fields": "paperId,title,abstract,authors,year,url,externalIds,isOpenAccess,openAccessPdf"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
)

if response.status_code == 200:

    data = response.json()

    papers = []

    for paper in data["data"]:
        papers.append({
            "paper_id": paper.get("paperId"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "authors": [ author["name"] for author in paper.get("authors", [])],
            "abstract": paper.get("abstract"),
            "url": paper.get("url"),
            "doi": (paper.get("externalIds") or {}).get("DOI"),
            "is_open_access": paper.get("isOpenAccess"),
            "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url")
        })

    with open("retrieved_papers.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Retrieved {len(papers)} papers.")
    print("Saved to retrieved_papers.json")

else:
    print("Error:", response.status_code)
    print(response.text)