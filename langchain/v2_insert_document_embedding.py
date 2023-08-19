import boto3
import json
import numpy as np
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


pdf_json_file_name = 'output.json'

with open(pdf_json_file_name, 'r') as f:
    data = json.load(f)

from typing import List, Tuple

collection_name = "documents"

loader = TextLoader("./Selected_Document_1.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


connection_string = PGVector.connection_string_from_db_params(
    driver = "psycopg2",
    port = "5432",
    user = "postgres",
    password = "password",
    host = "database-1.cb1rukuxcy0o.us-east-1.rds.amazonaws.com",
    database = "docembedding"
)

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/x-text"
    accepts = "application/x-text"

    def transform_input(self, payload: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(payload)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        embeddings = np.array(embeddings).reshape(1, -1)

        # normalization before inner product
        embeddings = normalize(embeddings, axis=1,norm='l2')
        return embeddings


content_handler = ContentHandler()

embeddings = SagemakerEndpointEmbeddings(
    # endpoint_name="endpoint-name",
    # credentials_profile_name="credentials-profile-name",
    endpoint_name="jumpstart-dft-robertafin-large-wiki-uncased",
    region_name="us-east-1",
    content_handler=content_handler,
)

db = PGVector.from_documents(
     embedding=embeddings,
     documents=docs,
     collection_name=collection_name,
     connection_string=connection_string
)

query = "What do some of the positive reviews say?"
docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.content)
    print("-" * 80)
