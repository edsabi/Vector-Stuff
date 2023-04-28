import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
openai_API = 'OPENAIKEY'

pinecone.init(
    api_key='PINECONEKEY',
    environment='ENVIRONMENT'
)

index_name = "index"

pdf_folder = "PATH"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

texts = []
for pdf_file in pdf_files:
    try:
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts += [t.page_content for t in text_splitter.split_documents(data)]
    except Exception as e:
        print(f"Error loading PDF file '{pdf_file}': {e}")

embeddings = OpenAIEmbeddings(openai_api_key=openai_API)
docsearch = Pinecone.from_texts(texts, embeddings, index_name=index_name)

while True:
    query=input("Input Query: ")
    docs= docsearch.similarity_search(query, include_metadata=True)


    llm = OpenAI(temperature=0, openai_api_key=openai_API)

    chain = load_qa_chain(llm, chain_type="stuff")
    blah= chain.run(input_documents=docs, question=query)
    print(blah)
