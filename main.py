#libraries import
from functions import make_FAISS_db, prompt_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

# QA class definition
class QASystem:
    def __init__(self, llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", max_new_tokens=500):
        """__init__ function initialize the system with given parameters

        Keyword Arguments:
            llm_model_name {str} -- the model for generating responses, I used tiny Llama from HuggingFace (default: {"TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
            embedding_model_name {str} -- the model for generating embeddings, I used popular all-MiniLM-L6-v2 from HuggingFace (default: {"sentence-transformers/all-MiniLM-L6-v2"})
            max_tokens {int} -- to control the maximum amount of information llm returns (default: {500})
        """
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.database = make_FAISS_db(embedding_model_name)
        self.max_new_tokens = max_new_tokens
        self.llm = self.get_base_llm()
        self.prompt = prompt_template()
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_base_llm(self):
        """get_base_llm generates pipeline of the future text generation. Here I used repetition_penalty = 1.2, other parameters are basic.

        Returns:
            HuggingFacePipeline -- ready pipeline for text generation
        """
        print("Creating model pipeline...")
        self.pipeline =  pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            repetition_penalty=1.2,
            return_full_text=True,
            max_new_tokens=self.max_new_tokens
        )
        return HuggingFacePipeline(pipeline=self.pipeline)

    def answer(self, user_query, number_of_articles=1):
        """answer function generate the response from llm with RAG option

        Arguments:
            user_query {str} -- user question.
            number_of_articles {int} -- number of articles to generate context

        Returns:
            dict -- answer from llm with context.
        """
        print("LLM answers...")
        retriever = self.database.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 1}
        )
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.llm_chain
        )
        response = rag_chain.invoke(user_query)
        return response

if __name__ == '__main__':
    # set the parametrs
    llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    max_new_tokens = 500
    number_of_articles_for_context = 1
    
    # test the system
    test_model = QASystem(llm_model_name=llm_model_name, embedding_model_name=embedding_model_name, max_new_tokens=max_new_tokens)
    test_query = "What is linear regression?"
    answer = test_model.answer(user_query=test_query, number_of_articles=number_of_articles_for_context)
    print('Question:\n', answer['question'])
    print('Context:\n', answer['context'][0].to_json()['kwargs']['metadata']['Text'][:200] + '...')
    print('\n', answer['text'].split('[/INST]')[1].strip())