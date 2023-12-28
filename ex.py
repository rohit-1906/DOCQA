from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
import uuid

loader = PyPDFLoader("budget_speech 1.pdf")
docs = loader.load()
text_splitter = TokenTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

parent_text_splitter = TokenTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
child_text_splitter =  TokenTextSplitter(
chunk_size=300,
            chunk_overlap=100,
        )
sub_child_text_splitter =  TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )
parent_docs = parent_text_splitter.split_documents(docs)
        

doc_ids = [str(uuid.uuid4()) for _ in parent_docs]

child_docs = []
for i, doc in enumerate(parent_docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata["doc_id"] = _id
    child_docs.extend(_sub_docs)

sub_child_docs = []
for i, doc in enumerate(parent_docs):
    _id = doc_ids[i]
    _sub_docs = sub_child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata["doc_id"] = _id
    child_docs.extend(_sub_docs) 
embeddings = OpenAIEmbeddings()
vectorstore = FAISS(embedding_function=embeddings)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryByteStore(),
    search_kwargs={"k": 2},
)

retriever.vectorstore.add_documents(child_docs)
retriever.vectorstore.add_documents(sub_child_docs)
retriever.docstore.mset(list(zip(doc_ids, parent_docs)))

l = input()
print(retriever.get_relevant_documents(l))

