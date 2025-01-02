import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# upstage models
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
# db 이름
index_name = "samsung-pdfs"  # Changed index name to reflect multiple PDFs
pdf_paths = [ 
    "삼성물산_패션부문_전문직.pdf",
    "에스원.pdf",
    "제일기획.pdf",
    "삼성물산_패션부문_신입.pdf",
    "삼성물산_리조트부문.pdf",
    "삼성물산_건설부문.pdf",
    "삼성E&A.pdf",
    "삼성중공업.pdf",
    "삼성증권.pdf",
    "삼성카드.pdf",
    "삼성SDS.pdf",
    "삼성전기.pdf",
    "삼성SDI.pdf",
    "삼성디스플레이.pdf",
    "삼성전자_DX부문.pdf",
    "삼성전자_DS부문.pdf",
]


# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start")

all_splits = []
for pdf_path in pdf_paths:
    print(f"Processing: {pdf_path}")
    document_parse_loader = UpstageDocumentParseLoader(
        pdf_path,
        output_format='html',
        coordinates=False)

    docs = document_parse_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    all_splits.extend(splits)  # Accumulate splits from all PDFs

print("Embedding and storing vectors...")
PineconeVectorStore.from_documents(
    all_splits, embedding_upstage, index_name=index_name
)
print("end")