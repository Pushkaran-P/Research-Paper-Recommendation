#  Research Paper Recommendation System Using Large Language Models

Note: This notebook is designed to run in colab free tier gpu, I am limited by the technology( hardware and money ) of my time to use better embedding models, llm's( Can't fit all these in a 6gb gpu :( ), containerizing in docker and deploying in ECS(AWS). You will find some techniques I've used here(some are dumb and can be optimised {See "Future"}) that are only seen in documentation and not in the countless blogs/Yt vids on this topic, all not to stick to conformity and to achieve a sense of fulfilling learning this topic.

## Future
- Document the code
- Make it much more interactive by streamlit/fastapi/flask in colab if possible
- Add video
- Try using CSVLoader but first check if you can do it with langchain_community.vectorstores.chroma.Chroma functions (to avoid embed_query error)
- Include persistent database to store chat history (SQL will do)
- Try to use bigger models and loading them in lower precision using BitsandBytes or if the model provides that option
- Finetuning on said bigger model or dolly ( Use other notebook as reference)

## Task
- Objective: The goal of this project is to develop a system that allows users to enter a particular topic and receive recommendations for research papers related to it. By web scraping from IEEE and other research paper websites, we aim to provide users with relevant and up-to-date research in their field of interest.

- Methodology: The project will involve the use of TF-IDF and similarity matching between the title and abstract of papers to recommend similar papers to a particular user input. The user can also interact with a chatbot to ask questions from these research papers.


## How I did it   

- Scrape data from research papers in IEEE websites (If you decide to run the mentioned code in Notebook 1, the process takes about 10 hours so I have kept the extracted data seperately in the repo.) The data is the title, abstract, citations, authors, year of publishing.

### TF-IDF
- Clean and preprocess the data (The usual stopwords removal, lemetization, special characters, lowercase etc...). Concatinated Title and Abstract
- TF-IDF matrix for all the texts, cosine similarity for... similarity :) and a simple while loop for question and relevant docs

### LLM
- TF-IDF approach is bad cause, no semantic and syntactic relations + unigram + outdated
- Enter the one topic people can't stop talking about, Generative AI specifically Large Language Models.
- I've used dolly-v2-3b as the llm engine ( Falcon 7B is another great option to run in free tier ), Chroma as the Vector Database, and bge-small-en-v1.5 as the vector embedding model. These are my personal preferences but pick your poision.
- Alongside this I've also included to get context from previous conversations.

## Questions
- Why not use chromadb seperately? RAG with langchain expects the retreiver( db.as_retriever() ) to have a embed_query() call, but when I tried chromadb seperately I found it does not have that call, maybe it works in other databases.
- Why have you copy pasted direct code from langchain? Bad and confusing documentation by langchain and very few blogs to look into this.
