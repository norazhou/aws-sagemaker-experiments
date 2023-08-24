from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json
from typing import Dict, List

connection_string = PGVector.connection_string_from_db_params(
    driver = "psycopg2",
    port = PORT,
    user = USER,
    password = PASSWORD,
    host = DB_HOST,
    database = DB_NAME
)

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/x-text"
    accepts = "application/x-text"

    def transform_input(self, payload: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(payload)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        print(output)
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        embeddings = np.array(embeddings).reshape(1, -1)

        # normalization before inner product
        embeddings = normalize(embeddings, axis=1,norm='l2')
        return embeddings


content_handler = ContentHandler()

#embeddings = OpenAIEmbeddings()
#embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = SagemakerEndpointEmbeddings(
    # endpoint_name="endpoint-name",
    # credentials_profile_name="credentials-profile-name",
    endpoint_name="jumpstart-dft-robertafin-large-wiki-uncased",
    region_name="us-east-1",
    content_handler=content_handler,
)

store = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    #collection_name="fictitious_hotel_reviews",
    distance_strategy=DistanceStrategy.COSINE
)

retriever = store.as_retriever(search_kwargs={"k": 1})

retriever.get_relevant_documents(query='What are the names of executive directors and senior managemnt?')
