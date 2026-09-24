# Embedding
# * Project Overview

This project demonstrates how sentence embeddings can be generated using a pre-trained Sentence Transformer model and how the semantic similarity between sentences can be calculated using Cosine Similarity.

The project uses the `all-mpnet-base-v2` model from Sentence Transformers to convert sentences into numerical vectors called embeddings.

These embeddings can then be compared to identify sentences that have similar meanings.

---

# * Objectives

The main objectives of this project are:

* Understand the concept of sentence embeddings
* Convert text sentences into numerical vector representations
* Use a pre-trained Sentence Transformer model
* Calculate similarity between sentences
* Identify semantically similar sentences using Cosine Similarity
* Understand how embeddings are used in Natural Language Processing (NLP)

---

# * What are Embeddings?

An embedding is a numerical representation of text.

Instead of representing a sentence simply as words, an embedding converts the sentence into a vector of numbers that captures information about its meaning and context.

For example:

```
"I enjoy coding in Python."
```

can be converted into a numerical vector such as:

```
[0.021, -0.134, 0.456, ...]
```

The actual vector contains many numerical values.

Sentences with similar meanings tend to have embeddings that are closer to each other in the vector space.

---

# * What is Sentence Embedding?

A sentence embedding represents an entire sentence as a fixed-size numerical vector.

In this project, the following model is used:

```python
SentenceTransformer("all-mpnet-base-v2")
```

The model converts each sentence into an embedding vector.

The `all-mpnet-base-v2` model produces embeddings with 768 dimensions.

---

# * Cosine Similarity

Cosine similarity measures how similar two vectors are based on the angle between them.

The formula is:

```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

The value generally ranges from:

| Value | Meaning |
|-------|---------|
| 1 | Very similar |
| 0 | Unrelated |
| -1 | Opposite direction |

In this project, sentences with a similarity score greater than **0.7** are displayed as similar.

---

# * Technologies Used

* Python
* Sentence Transformers
* Scikit-learn
* NumPy (used internally by the libraries)
* NLP / Natural Language Processing

---

# * Installation

## Clone the Repository

```bash
git clone <repository-url>
```

## Navigate to the Project Folder

```bash
cd embedding
```

## Install Required Libraries

```bash
pip install sentence-transformers scikit-learn
```

---

# * Project Structure

```text
Embedding/
│
├── embedding.py
└── README.md
```

---

# * Code Explanation

## Step 1: Import Libraries

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

`SentenceTransformer` is used to generate sentence embeddings.

`cosine_similarity` is used to calculate the similarity between the generated embeddings.

## Step 2: Load the Pre-trained Model

```python
model = SentenceTransformer("all-mpnet-base-v2")
```

This loads the pre-trained `all-mpnet-base-v2` model.

The model has already learned semantic relationships from large amounts of text, allowing it to generate meaningful sentence embeddings.

## Step 3: Create Sentences

```python
sentences = [
    "I enjoy coding in Python.",
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "The weather is very hot today.",
    "It is raining heavily outside.",
    "I went to college this morning.",
    "My college has many computer science students.",
]
```

The sentences contain different topics such as:

* Python programming
* Weather
* College

This allows us to observe how embeddings capture semantic relationships.

## Step 4: Generate Embeddings

```python
embeddings = model.encode(sentences)
```

The `encode()` function converts every sentence into a numerical vector.

Since `all-mpnet-base-v2` produces 768-dimensional embeddings, each sentence is represented by a vector containing 768 numerical values.

## Step 5: Display Number of Sentences and Embedding Dimension

```python
print("Total number of sentences:", len(sentences))
print("Embedding dimension:", len(embeddings[0]))
```

This displays:

* Total number of sentences
* Number of values in each embedding vector

For this project:

```
Total number of sentences: 7
Embedding dimension: 768
```

## Step 6: Display Embeddings

```python
for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Embedding:", embeddings[i])
```

This prints each sentence along with its numerical embedding.

The embedding contains 768 numerical values representing the semantic information of the sentence.

## Step 7: Calculate Cosine Similarity

```python
similarity = cosine_similarity(embeddings)
```

This calculates the cosine similarity between every pair of sentence embeddings.

The result is a similarity matrix. For example:

```
         S1     S2     S3
S1     1.00   0.85   0.78
S2     0.85   1.00   0.82
S3     0.78   0.82   1.00
```

A higher value indicates greater semantic similarity.

## Step 8: Find Similar Sentences

```python
if similarity[i][j] > 0.7:
```

Only sentence pairs having a cosine similarity greater than 0.7 are displayed.

The similarity value is formatted using:

```python
f"{similarity[i][j]:.4f}"
```

which displays the result up to four decimal places.

---

# * Project Workflow

```
Input Sentences
      ↓
Pre-trained Sentence Transformer
      ↓
Generate Sentence Embeddings
      ↓
768-Dimensional Vectors
      ↓
Calculate Cosine Similarity
      ↓
Compare Similarity Scores
      ↓
Display Similar Sentences
```

---

# * Example

The following sentences have similar meanings:

* "I enjoy coding in Python."
* "I love programming in Python."
* "Python is my favorite programming language."

Their embeddings are expected to have relatively high cosine similarity because they are semantically related to Python programming.

On the other hand:

* "I enjoy coding in Python."
* "The weather is very hot today."

are about different topics, so their similarity is expected to be lower.

---

# * Real-World Applications

Sentence embeddings are widely used in modern NLP applications.

**Semantic Search** — Search engines can compare the meaning of a user's query with documents instead of only matching exact keywords.

**Chatbots** — Embeddings can help chatbots identify questions or messages with similar meanings.

**Document Similarity** — Documents can be converted into embeddings and compared to find similar documents.

**Question Matching** — Frequently asked questions can be matched with previously stored answers.

**Recommendation Systems** — Embeddings can help recommend content based on semantic similarity.

**Duplicate Detection** — Similar articles, questions, or text documents can be identified using embedding similarity.

---

# * Advantages

* Captures the semantic meaning of sentences
* More effective than simple keyword matching for many NLP tasks
* Uses a pre-trained model
* Easy to implement using Sentence Transformers
* Useful for semantic search and recommendation systems
* Can compare sentences even when they use different words with similar meanings

---

# * Limitations

* Embeddings can require significant computational resources for large datasets.
* Similarity thresholds such as 0.7 may need to be adjusted depending on the application.
* Similarity scores do not always perfectly represent human judgment.
* The quality of embeddings depends on the selected model and the type of text.

---

# * Future Improvements

This project can be extended by:

* Adding a user interface using Streamlit
* Allowing users to enter their own sentences
* Creating a sentence similarity search system
* Visualizing embeddings using PCA or t-SNE
* Building a semantic search engine
* Comparing multiple embedding models
* Storing embeddings in a vector database such as FAISS or ChromaDB

---

# * Key Concepts Learned

Through this project, I learned:

* What text embeddings are
* What sentence embeddings are
* How pre-trained Transformer models generate embeddings
* How all-mpnet-base-v2 can be used for sentence representation
* How cosine similarity compares vectors
* How semantic similarity differs from simple keyword matching
* How embeddings can be applied to real-world NLP applications

---

# * Author

**D. Hasini**

B.Sc. Computer Science with AI, Semester III

---

# * Conclusion

This project demonstrates the basic workflow of text embedding and semantic similarity using Sentence Transformers.

By converting sentences into numerical vectors and comparing those vectors using cosine similarity, computers can work with the semantic meaning of text rather than relying only on individual words.

This concept forms an important foundation for applications such as semantic search, recommendation systems, question matching, chatbots, and Retrieval-Augmented Generation (RAG).
