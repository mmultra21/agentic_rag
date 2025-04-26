from serpapi import GoogleSearch

search = GoogleSearch({
    "q": "agentic RAG",
    "api_key": "ca0b6c7cbbb5d8d254a4573e746e4308e3767f96ca4c3a5b6805dbc6a060d68e"
})

results = search.get_dict()
print(results.get("organic_results", []))