import openai
import pandas as pd
import numpy as np
import pickle
import os

import torch
#from transformers import GPT2TokenizerFast, GPT2LMHeadModels #https://huggingface.co/docs/transformers/model_doc/gpt2

from transformers import GenerationConfig, GPT2TokenizerFast, AutoTokenizer, GPT2LMHeadModel, BloomModel, BloomTokenizerFast

#from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from typing import List, Dict, Tuple
import requests
import json
import nltk
from urllib.request import urlopen

openai.api_key = "TU APP KEY"


MODEL_NAME = "curie"
MODEL_NAME = "curie-instruct-beta"
MODEL_NAME = "davinci"
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "

COMPLETIONS_MODEL = "text-davinci-003"
COMPLETIONS_MODEL = f"text-{MODEL_NAME}-003"
COMPLETIONS_API_PARAMS = {
    "model": COMPLETIONS_MODEL,
}

#os.environ['TOKENIZERS_PARALLELISM'] = True

'''
GPT2 - Numbers of attentions layers:
gpt2: 12
gpt2-medium: 24
gpt2-large: 36
gpt2-xl: 48
'''
model_base = "gpt2"
#model_base = "gpt2-xl"



'''
BLOOM - Numbers of attentions layers:
bloom-560m
bloom-1b1
bloom-1b7
bloom-3b
bloom-7b1
bloom (176B parameters)
'''
#model_base = "bigscience/bloom-560m"
#model_base = "bigscience/bloom"


#GPT2 Model
# get the tokenizer for the pre-trained LM you would like to use
tokenizer = GPT2TokenizerFast.from_pretrained(model_base)
#tokenizer = AutoTokenizer.from_pretrained(model_base)    
#tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

# instantiate a model (causal LM)
model = GPT2LMHeadModel.from_pretrained(model_base,
                                    output_scores=True,
                                    pad_token_id=tokenizer.eos_token_id)
model_home = "./model/"
# Download configuration from huggingface.co and cache.
generation_config = GenerationConfig.from_pretrained(model_base)

# Our config was saved using *save_pretrained('./test/saved_model/')*
#generation_config.save_pretrained(model_home)
#generation_config = GenerationConfig.from_pretrained(model_home)

# Configuration names to your generation configuration file
generation_config.save_pretrained(model_home, config_file_name="config.json")
generation_config = GenerationConfig.from_pretrained(model_home, "config.json")

# If you'd like to try a minor variation to an existing configuration, you can also pass generation
# arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
generation_config, unused_kwargs = GenerationConfig.from_pretrained(
    "gpt2", top_k=1, foo=False, return_unused_kwargs=True
)
generation_config.top_k

#BLOOM model
#tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
#tokenizer = BloomTokenizerFast.from_pretrained(model_base) 
#model = BloomModel.from_pretrained(model_base,
#                                    output_scores=True,
#                                    pad_token_id=tokenizer.eos_token_id)

# inspecting the (default) model configuration
# (it is possible to created models with different configurations)
print(model.config)

separator_len = len(tokenizer.tokenize(SEPARATOR))

def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text)
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str, actual_file) -> Dict[Tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    df1 = pd.read_csv(actual_file)
    new_df = df1.merge(df, left_index=True, right_index=True)
    max_dim = max([int(c) for c in new_df.columns if c != "title" and c != "heading" and c != 'content' and c != 'tokens'])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in new_df.iterrows()
    }, new_df


def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

from typing import Dict, List, Tuple

def order_document_sections_by_query_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, document_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, document_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, (_, section_index) in most_relevant_document_sections:
        # Add contexts until we run out of space.

       ## This changed
        document_section = df[df.heading == section_index]

        for i, row in document_section.iterrows():
            chosen_sections_len += row.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + row.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

# Here, Let's create a function to answer the query with the retrieved document context:
def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False, temperature=0, max_length=500, choice_model=""
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    COMPLETIONS_API_PARAMS['temperature'] = temperature
    COMPLETIONS_API_PARAMS['max_tokens'] = max_length
    
    if choice_model != "":
        COMPLETIONS_API_PARAMS['model'] = choice_model

    print("choice_model::::",choice_model)

    '''
    #Advanced
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,

    COMPLETIONS_API_PARAMS['top_p'] = top_p
    COMPLETIONS_API_PARAMS['frequency_penalty'] = frequency_penalty
    COMPLETIONS_API_PARAMS['presence_penalty'] = presence_penalty
    '''  

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

## Earnings Call API ##

def earnings_summary(ticker, quarter, year):
    apikey = 'c321d1c2d7401e5f0029626a16fd80d5'
    url = requests.get(f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?quarter={quarter}&year={year}&apikey={apikey}")
    try:
    # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen

    def get_jsonparsed_data(url):
        """
        Receive the content of ``url``, parse it as JSON and return the object.

        Parameters
        ----------
        url : str

        Returns
        -------
        dict
        """
        response = urlopen(url)
        data = response.read().decode("utf-8")
        return json.loads(data)

    df = get_jsonparsed_data(url.url)

    df_2 = df[0]['content']
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

    def split_text(text, length):
        # tokenize the text
        tokens = word_tokenize(text)
        # calculate the number of rows needed
        rows = len(tokens) // length + (len(tokens) % length != 0)
        # split the tokens into rows
        split_tokens = [tokens[i:i+length] for i in range(0, len(tokens), length)]
        # join the tokens in each row back into a string
        split_text = [" ".join(row) for row in split_tokens]
        return split_text

    # example usage
    text = df_2
    length = 100
    result = split_text(text, length)

    # Convert the result to a DataFrame
    df_3 = pd.DataFrame(result, columns=["content"])
    df_3['title'] = str(df[0]['symbol']) + ' Q' + str(df[0]['quarter']) + ' ' + str(df[0]['year'])
    df_3['heading'] = str(df[0]['date'])
    df_3['tokens'] = df_3['content'].str.len() / 4
    # Write the DataFrame to a CSV file
    df_3.to_csv('prepared_ec.csv', index=False)

'''
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)
 
completion_with_backoff(model="text-davinci-003", prompt="Once upon a time,")
'''