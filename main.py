from utils import make_FAISS_db, prompt_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import os
import argparse
from dotenv import load_dotenv

load_dotenv()
try:
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
except:
    raise ValueError(
        "Cohere API key is not set. Please set the COHERE_API_KEY environment variable."
    )


# QA class definition
class QASystem:
    def __init__(
        self,
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_new_tokens=500,
    ):
        """__init__ function initialize the system with given parameters

        Keyword Arguments:
            llm_model_name {str} -- the model for generating responses, I used tiny Llama from HuggingFace (default: {"TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
            embedding_model_name {str} -- the model for generating embeddings, I used popular all-MiniLM-L6-v2 from HuggingFace (default: {"sentence-transformers/all-MiniLM-L6-v2"})
            max_tokens {int} -- to control the maximum amount of information llm returns (default: {500})
        """
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.database = make_FAISS_db(embedding_model_name)
        print("Creating model pipeline...")
        self.max_new_tokens = max_new_tokens
        self.llm = self.get_base_llm()
        self.prompt = prompt_template()
        self.llm_chain = LLMChain(
            llm=self.llm, prompt=self.prompt, output_parser=StrOutputParser()
        )

    def get_base_llm(self):
        """get_base_llm generates pipeline of the future text generation. Here I used repetition_penalty = 1.1, other parameters are basic.

        Returns:
            HuggingFacePipeline -- ready pipeline for text generation
        """
        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=self.max_new_tokens,
        )
        return HuggingFacePipeline(pipeline=self.pipeline)

    def answer(self, user_query, number_of_articles=1):
        """answer function generate the response from llm with RAG option.
        Additionally uses Cohere Rerank to rerank retrieved articles and find the most relevant one.

        Arguments:
            user_query {str} -- user question.
            number_of_articles {int} -- number of articles to generate context

        Returns:
            dict -- answer from llm with context.
        """
        print("LLM answers, it could take a min...")
        self.naive_retriever = self.database.as_retriever(
            search_type="similarity",
            search_kwargs={"k": number_of_articles},
        )

        compressor = CohereRerank(top_n=1)
        self.cohere_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.naive_retriever
        )

        rag_chain = {
            "context": self.cohere_retriever,
            "question": RunnablePassthrough(),
        } | self.llm_chain
        response = rag_chain.invoke(user_query)
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", type=str)
    parser.add_argument(
        "--embedding_model_name", default="sentence-transformers/all-MiniLM-L6-v2", type=str
    )
    parser.add_argument("--max_new_tokens", default=1000, type=int)
    parser.add_argument("--number_of_articles_for_context", default=10, type=int)
    parser.add_argument("--test_query", default="What is linear regression?", type=str)
    args = parser.parse_args()

    # set the parametrs
    llm_model_name = args.llm_model_name
    embedding_model_name = args.embedding_model_name
    max_new_tokens = args.max_new_tokens
    number_of_articles_for_context = args.number_of_articles_for_context
    test_query = args.test_query

    # test the system
    test_model = QASystem(
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        max_new_tokens=max_new_tokens,
    )
    answer_with_RAG = test_model.answer(
        user_query=test_query, number_of_articles=number_of_articles_for_context
    )
    print("Question:\n", answer_with_RAG["question"])
    print("Articles from naive retriever output:\n")
    naive_retriever = test_model.naive_retriever.invoke(test_query)
    for i in range(len(naive_retriever)):
        print("\nTitle", i + 1, ": ", naive_retriever[i].to_json()["kwargs"]["metadata"]["Title"])
    print("\nArticles from Cohere rerank retriever used for context output:\n")
    cohere_retriever = test_model.cohere_retriever.invoke(test_query)
    for i in range(len(cohere_retriever)):
        print("\n", cohere_retriever[i].to_json()["kwargs"]["metadata"])
        print("\n", cohere_retriever[i].to_json()["kwargs"]["page_content"])
    print("\nAnswer from the LLM:\n")
    print(answer_with_RAG["text"].split("[/INST]")[1].strip())
