from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

import os
from langchain_pinecone import Pinecone   
from langchain.embeddings import HuggingFaceEmbeddings






# Callbacks support token-wise streaming
def load_model() -> LlamaCpp:
    
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
    Model_path = "C:\LLAMA2Locally\model\llama-2-13b-chat.Q3_K_M.gguf"
    # Make sure the model path is correct for your system!
    Llama_model: LlamaCpp = LlamaCpp(
        model_path=Model_path,
        temperature=0.75,
        max_tokens=20,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True # Verbose is required to pass to the callback manager
    )
    return Llama_model


# llm = load_model()

# model_prompt: str = """
# Question: what is metaverse
# """

# response: str = llm(model_prompt)

def ragconfi():
    pinecone_api_key = "de6f6c4b-7e94-4e81-9fc2-072ba2b19bbc"
    pinecone_index_name = "docsindex"

    # Set the environment variables
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["PINECONE_INDEX_NAME"] = pinecone_index_name

    
    # load csv files
    loader = CSVLoader(file_path='C:\LLAMA2Locally\llm_models.csv')
    docs = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    
    index_name = "docsindex"
    
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return docsearch
    
    
    
def prompttemp():
    SYSTEM_PROMPT = """only understand the question and give answer according to the provided csv.
If you don't know the answer, just say that you don't know answer."""



    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"


    SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS


    instruction = """
    {context}

    Question: {question}
    """


    template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    
    return prompt
    
def retrivalchain(question):
    
    llm = load_model()
    docsearch = ragconfi()
    prompt = prompttemp()
    response_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    result = response_chain('question')
    return result['result']
    
res = retrivalchain("who is the owner of bert")
# result['result']
print(res)
