# Project overview
This is a Question-Answering program based on [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) llm from [HuggingFace](https://huggingface.co/). The project also involves RAG system based on [1300+ Towards DataScience Medium Articles Dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset) data set from kaggle. Additionally I added rerank method, which works as follows:
1. Get k(default 10) chunks from database
2. With the help of Cohere API, rerank this articles and select top_n(default 1) chunks relatively to the previously selcted chunks. And use only them for context of LLM query

This method suppose to help to get more meaningful and precise information from the LLM.

Retrieve Augmented Generation in my program works as follows:

![Schema of the program](Schema.png "Schema of the program")

# Set up and configuration
You can download the repository via the command
```
git clone https://github.com/DzmitryPihulski/LLM_question_and_answer_system_with_RAG.git
```
Install all the libraries by running, I used python version 3.11.4

```
pip install -r requirements.txt
```

You will also need a .env file in the folder with the project. In the .env file you need to place your Cohere API Key. You can create yours for free on theirs [website](https://dashboard.cohere.com/api-keys). In the file paste your key:

```

COHERE_API_KEY = "YOUR_API_KEY_FROM_THE_WEBSITE"
```

And after that run the main.py
It will generate a response for the test question.
If you want to change the question, you must change it in the main.py, "test_query" variable.