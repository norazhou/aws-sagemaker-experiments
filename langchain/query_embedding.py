#!/usr/bin/python

import psycopg2
import json
from sklearn.preprocessing import normalize
import boto3
import numpy as np

def query_vector(embedding):
    sql = "SELECT page_number, content FROM items ORDER BY embedding <#> '" + str(embedding.tolist()) + "' LIMIT 3;"
    conn = None
    vendor_id = None
    result = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(host="database-1.cb1rukuxcy0o.us-east-1.rds.amazonaws.com", 
                                database="docembedding",
                                user="postgres",
                                password="password")
        # create a new cursor
        cur = conn.cursor()
        # execute the QUERY statement
        cur.execute(sql, json.dumps(embedding.tolist()))
        result = cur.fetchall()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result

def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime", region_name='us-east-1')
    endpoint_name = "jumpstart-dft-robertafin-large-wiki-uncased"
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-text",
        Body=json.dumps(payload),
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

from sklearn.preprocessing import normalize

def parse_response(query_response):
    """Parse response and return the embedding."""
    model_predictions = query_response
    embeddings = model_predictions["embedding"]
    embeddings = np.array(embeddings).reshape(1, -1)

    # normalization before inner product
    embeddings = normalize(embeddings, axis=1,norm='l2')
    return embeddings


def query(query):

    if query == None:
        query = """
We retrieve information from database, and with to fact check the following statement, based on provided information.

Statement: management continuity for at least the three preceding financial years. In order words, top executive director team is consistent in previous 3 years, and do not change often.

Check if the statement is valid based on all executive director's background. In addtion, please summarise why it is VALID or NOT based on the statement. The answer should not go beyond 200 words.
"""


    query_response = query_endpoint(query)
    embedding = parse_response(query_response).squeeze()
    data = query_vector(embedding)


    #for content in data:
        #print(content)
        #print("\n\n\n")

    #print(data)

    return data

