#!/usr/bin/python

import psycopg2
import json
from sklearn.preprocessing import normalize
import boto3
import numpy as np

endpoint_name = "jumpstart-dft-mx-tcembedding-robertafin-large-uncased"

def chunk_words(sequence, chunk_size):
    sequence = sequence.split()
    return [' '.join(sequence[i:i+chunk_size]) for i in range(0, len(sequence), chunk_size)]

def query_vector(embedding):
    sql = "SELECT page_number, content FROM v3_items ORDER BY embedding <=> '" + str(embedding) + "' LIMIT 10;"
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
        cur.execute(sql)
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
    embeddings = []
    client = boto3.client("sagemaker-runtime", region_name="us-east-1")
    chunk_payload = chunk_words(payload, 400)
    for i, chunk in enumerate(chunk_payload):
        #print("Chunk ",i)
        #print("Content ",chunk)
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/x-text",
            Body=json.dumps(chunk),
        )    
        response = response["Body"].read().decode("utf8")
        response = json.loads(response)
        embeddings_chunk = response["embedding"]
        embeddings.append(embeddings_chunk)
    return embeddings

from sklearn.preprocessing import normalize

def parse_response(query_response):
    """Parse response and return the embedding."""
    embeddings = np.array(query_endpoint(query_response))
    #avg_embeddings = np.mean(embeddings, axis=0)
    # try max pooling of embedding vector
    avg_embeddings = np.max(embeddings, axis=0)

    avg_embeddings = avg_embeddings.reshape(1, -1)
    # normalization before inner product
    avg_embeddings = normalize(avg_embeddings, axis=1, norm='l2')
    return np.squeeze(avg_embeddings)


def query(query):

    query_embedding = parse_response(query).tolist()
    #print(query_embedding)
    data = query_vector(query_embedding)
    #print(data)


    #for content in data:
    #    print(content)
    #    print("\n\n\n")

    return data

query("what are responsibilities of zhang feng?")
query("whether this statement is valid:  management continuity for at least the three preceding financial years. In order words, top director team is consistent in previous 3 years, and do not change often")
