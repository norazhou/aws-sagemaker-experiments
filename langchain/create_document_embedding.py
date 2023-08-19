import boto3
import json
import numpy as np

endpoint_name = "jumpstart-dft-robertafin-large-wiki-uncased"

def extract_pageindex(origin_str, sub1, sub2):
    # getting index of substrings
    idx1 = origin_str.index(sub1)
    idx2 = origin_str.index(sub2)

    # length of substring 1 is added to
    # get string from next character
    res = origin_str[idx1 + len(sub1): idx2]

    # printing result
    return res

from pathlib import Path

pathlist = Path("documents/extractedpages").glob('**/*.pdf')
#pathlist = ['extractedpages/document-page560.pdf','extractedpages/document-page500.pdf']
output = []
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    file=open(path_in_str,"r")
    payload = file.read()
    file.close()

    page_index = extract_pageindex(path_in_str, "document-page", ".pdf")
    data = {}
    data["content"] = payload
    data["page_number"] = page_index
    data["pdf_file_path"] = path_in_str
    output.append(data)

with open('output.json', 'w') as outfile:
    json.dump(output, outfile, indent=4)

pdf_json_file_name = 'output.json'

with open(pdf_json_file_name, 'r') as f:
    data = json.load(f)

def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime", region_name='us-east-1')
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
    return embeddings.tolist()

for page_index, page in enumerate(data):
    if page_index % 30 == 0:
        print(f'Processing page {page_index}')
    query_response = query_endpoint(data[page_index]["content"])
    data[page_index]["embedding"] = parse_response(query_response)[0]

output_file_name = 'embedding.json'
with open(output_file_name, 'w') as f:
    # Use json.dump to write pdfText to the file
    json.dump(data, f, indent=4)


