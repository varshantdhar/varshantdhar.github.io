---
title:  "RAG Foundation - Extract, Chunk, Embed and Retrieve"
date:   2025-06-06
mathjax: true
categories:
    - blog
tags: 
    - RAG
    - LangChain
    - LLM
    - Prompt
    - LangGraph
---

Building the RAG Foundation – Extract, Chunk, Embed, Retrieve. These steps are foundational to any RAG (retrieval-augmented generation) system because they prepare unstructured data for efficient semantic querying and LLM-based reasoning.

### Step 1: Extracting from PDF
In real-world workflows, academic documents are rarely clean HTML or Markdown — they're often scanned or typeset PDFs. We need to extract readable, clean text from these formats. I chose `pdfplumber` for its reliable layout handling and ability to preserve mathematical structure like equation indentation and paragraph flow.

```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() or ""
```

This gives us one large text blob per document, which forms the base for downstream chunking. LaTeX content is preserved in its raw form — we’ll address how to handle equations more intelligently in later stages.

#### Text Cleaning During PDF Extraction

PDFs — especially scanned math textbooks — often contain inconsistent spacing and line breaks that corrupt the logical structure of paragraphs and formulas. I added a `clean_text()` function to fix issues such as:

- Removing random newlines
- Breaking up merged words like `thisisasentence` → `this is a sentence`
- Stripping excessive whitespace

This cleaning happens **before chunking**, which ensures that each chunk preserves logical continuity — especially critical for mathematical derivations.


### Step 2: Chunking for RAG
Raw text is too large and unstructured for LLM input. LLMs typically have token limits (e.g., 4K–8K tokens), and sending the entire textbook to the model would be inefficient and error-prone. To solve this, we split text into smaller, overlapping segments — called chunks — which act as self-contained contexts for retrieval.

I used LangChain’s RecursiveCharacterTextSplitter, which is optimized to preserve semantic meaning by splitting at logical separators:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
```
The overlap is key for math — it ensures formulas, definitions, or proofs don’t get sliced apart across chunks. Chunking is an essential preprocessing step that bridges document structure and vector-based search.

Alternatives to the method we put in place are

- Fixed window sliding: Fast, but slices formulas and paragraphs mid-thought
- Sentence-based chunking: May be too granular for derivations/proofs
- Markdown/Heading-based chunking: Great for structured docs, but academic PDFs rarely have clean headers
- Semantic chunking (e.g., TextTiling): tries to segment a document where the topic shifts, not just where a sentence or paragraph ends.

### Step 3: Embedding with Sentence Transformers
To retrieve chunks relevant to a user query, we need a way to compare their meanings. This is where embeddings come in — high-dimensional vector representations that encode semantic similarity. I used `sentence-transformers/all-MiniLM-L6-v2` to convert each chunk into a 384-dimensional vector:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
```
Embeddings allow us to compute cosine similarity between a user query and each chunk, enabling relevance-ranked retrieval. Traditional search (e.g., TF-IDF, keyword search, BM25) fails when:
- The query uses synonyms or paraphrasing.
- The content uses math/technical language where keyword overlap is low.

Semantic embeddings let us compare meaning, not just text.

### Step 4: Store in FAISS

We store all the embeddings in a FAISS index — an optimized library for fast similarity search. This allows us to perform vector-based nearest neighbor lookups across potentially thousands of chunks, all in milliseconds:

```python
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "vectorstore/faiss.index")
```
In addition to the index, we also save the original chunks using `pickle`, so that when a query returns a match, we can return the corresponding text directly. Think of this as building a searchable library of ideas and definitions, where every entry is semantically indexed.

We use the brute-force IndexFlatL2 which is a brute force indexing method that computes similarities with all the other vectors in the index. This is not scalable (like HNSW, Annoy or Quantization techniques) but would work for our pdfs. 

### Step 5: Querying the Vector Index

With our FAISS index and chunk mappings in place, we can now perform semantic queries. I built a lightweight retriever script that:

1. Loads the FAISS index and associated chunk text
2. Embeds a user query using the same model as above
3. Retrieves the top-k closest chunks
4. Displays them with similarity scores

This setup lets me ask open-ended questions like "What is a sufficient statistic?" and get back relevant textbook definitions, theorems, or examples from the embedded documents.

Example result:
```
Definition 3.9. A statistic T is minimal sufficient if T is sufficient, and for every sufficient statistic T˜ there exists a function f such that T = f(T˜)...
------------------------------------------------------------
```
This marks the first successful full-loop for retrieval in my RAG system.