import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import streamlit as st
# Import necessary modules from langchain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer, util
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# Set the LLM cache
set_llm_cache(InMemoryCache())

# Streamlit app title
st.title("PDF Summarizer, QA & Chat")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# if "chain" not in st.session_state:
#     st.session_state.chain = None

def qa_chain(llm):
    
    # template = """"
    prompt = PromptTemplate(
    input_variables=["docs", "question", "history"],
    template="""Given the following documents and historical context, provide an answer to the question:
    Documents: {docs}
    Historical Context: {history}
    Question: {question}
    
    Please refer to any relevant information from the previous response to answer the current question.
    """
)
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # Load QA chain
    chain1 = LLMChain(llm=llm,prompt=prompt)
    #print("chain")
    return chain1

if "llm" not in st.session_state and "chain" not in st.session_state:
    # Load OpenAI API key
    dotenv_path = "openai.env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Initialize OpenAI model
    OpenAIModel = "gpt-4"
    st.session_state.llm = ChatOpenAI(model=OpenAIModel, temperature=0)
    st.session_state.chain = qa_chain(st.session_state.llm)

    
def process_text(doc):
    # Split text into chunks
        text_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )
        chunks = text_splitter.split_documents(doc)
        #print(1)
        # Create embeddings and knowledge base using FAISS
        embeddings = OpenAIEmbeddings()

        knowledgeBase = FAISS.from_documents(chunks,embeddings)


        return knowledgeBase



# Function to process text from PDFs and create a knowledge base


# Function to calculate relevance score between question and response
def calculate_relevance_score(question, response):
    # Use Sentence Transformers to encode question and response
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    response_embedding = model.encode(response, convert_to_tensor=True)

    # Calculate cosine similarity for relevance score
    cosine_score = util.pytorch_cos_sim(question_embedding, response_embedding)
    relevance_score = cosine_score.item()

    return relevance_score

# Function to store conversation history
def store_conversation(user_query, assistant_response):

    st.session_state.conversation_history.append({"Question_number": len(st.session_state.conversation_history), "user_query": user_query, "assistant_response": assistant_response})
# Function to display conversation history in sidebar
def display_conversation_history():
    st.sidebar.subheader("Conversation History")
    for conv in st.session_state.conversation_history:
        st.sidebar.markdown(f"**User:** {conv['user_query']}")
        st.sidebar.markdown(f"**Assistant:** {conv['assistant_response']}")
        st.sidebar.markdown("---")

# Function to handle user interaction
def handle_user_interaction(user_question):
    knowledgeBase = st.session_state.knowledgeBase
    ret = knowledgeBase.as_retriever(
    search_kwargs={"k": 20}
)


    content_found = False
    relevant_responses = []
    compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ret
)
    # Search for relevant documents based on user's question
    docs = compression_retriever.get_relevant_documents(user_question)
    st.write(docs)
    # print(docs)
    # print("*"*100)
    if docs:
        content_found = True
        # Run Question Answering model to find response
        # OpenAIModel = "gpt-3.5-turbo"
        # llm = ChatOpenAI(model=OpenAIModel, temperature=0.1, openai_api_key=load_openai_api_key())
        # chain = qa_chain(llm, chain_type='stuff')
        response = st.session_state.chain.run(
            {"docs": docs, "question": user_question,"history":st.session_state.conversation_history}
        )
        # print(st.session_state.conversation_history)

        if response:
            # Calculate relevance score for the response
            relevance_score = calculate_relevance_score(user_question, response)
            relevant_responses.append({"response": response, "score": relevance_score})

    if not content_found:
        st.write("No relevant information found in the uploaded PDFs.")
    elif relevant_responses:
        # Get the most relevant response based on relevance score
        most_relevant = max(relevant_responses, key=lambda x: x["score"])
        # Store user query and assistant response in session state
        #st.session_state.messages.append({"role": "user", "content": user_question, "response": most_relevant['response']})
        # print( st.session_state.messages)
        # print("***************")
        assistant_response = f" \n\n{most_relevant['response']}"
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = assistant_response
            for chunk in assistant_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        store_conversation(user_question, assistant_response)
    else:
        st.write("I couldn't find any relevant information about your question in the uploaded PDFs.")

# Streamlit app
def main():
    # Interface for uploading files and displaying conversation history
    with st.sidebar.expander("Upload your PDF Documents"):
        pdf_files = st.sidebar.file_uploader(' ', type='pdf', accept_multiple_files=True)
        if pdf_files:
            st.session_state.uploaded_files = pdf_files
            if "knowledgeBase" not in st.session_state:
    # Process text from PDFs to create knowledge base if it doesn't exist
                doc = []
                for pdf in st.session_state.uploaded_files:
                    text = ""
                    i = 0
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                        i+=1
                        metadata = {
                            "filename": pdf.name,
                            "page_no": i
                        }
                        doc.append(Document(page_content=text, metadata=metadata))
                        
                # st.write(doc) 
                    # doc.extend(pdf_reader)   
                
                st.session_state.knowledgeBase = process_text(doc)
                # st.write(doc)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    display_conversation_history()  # Show conversation history in sidebar

    # User input
    if prompt := st.chat_input(
            "First, read all the documents. Then, find the relevant answer from the uploaded files. If the answer is in multiple files, please provide a single answer that is the most relevant."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        pdf_files = st.session_state.uploaded_files
        user_question = prompt.strip()

        if pdf_files and user_question:
            # st.session_state.messages = [message for message in st.session_state.messages if
            #                              message["role"] != "assistant"]
            # print(st.session_state.messages)
            handle_user_interaction(user_question)

            # for message in st.session_state.conversation_history:
            #     if message["role"] == "assistant":
            #         with st.chat_message("assistant"):
            #             st.markdown(message["content"])

# Execute the main function if this is the main script
if __name__ == '__main__':
    main()
