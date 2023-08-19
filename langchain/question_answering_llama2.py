from query_embedding import query
from langchain.chains import LLMChain
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
from langchain import PromptTemplate

def build_chain():
    endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b-f"
    region_name = "us-east-1"

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs" : [[
            {"role" : "user", "content" : prompt}]],
            "parameters" : {**model_kwargs}})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generation"]["content"]

    parameters = {
        "max_new_tokens": 2048,
        "temperature":0.3,
        "top_p": 0.9
    }

    content_handler = ContentHandler()

    sm_llm = SagemakerEndpoint(
        endpoint_name = endpoint_name,
        region_name = region_name,
        model_kwargs = parameters,
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler = content_handler,
    )


    prompt_template = """

Please help me to find the answer, based on the retrieved information and the user input question.

user input question:
{question}

retrieved information
{retrieved_information}

Please ensure the answer are relevant to retrieved information, if you could not find the answer, please wrote #IDONTKNOW exact words without any extrat word.

"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["retrieved_information", "question"]
    )

    llmchain = LLMChain(llm=sm_llm, prompt=prompt)
    return llmchain

def run_chain(chain, question: str, history=[]):
    answer = chain.run({
        'question': question,
        'retrieved_information': query(question)
    })
    
    result = {
        'answer': answer,
        'source': []
    }
    return result

if __name__ == "__main__":
    llmchain = build_chain()
    Question = "What are the names of executive directors and senior management?"
    result = run_chain(llmchain, Question)
    print(result)
