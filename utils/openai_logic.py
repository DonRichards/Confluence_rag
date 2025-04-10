# note which revision of python, for example 3.9.6
from datasets import load_dataset
from openai import OpenAI
# import openai
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import sys
import json 
import numpy as np
import gradio as gr

#Global variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# get embeddings
def get_embeddings(query, model_emb):
   try:
       if not query or not query.strip():
           raise ValueError("Empty query provided for embedding generation")
           
       if not model_emb:
           raise ValueError("No embedding model specified")
       
       embedding = openai_client.embeddings.create(input=query, model=model_emb)
       
       if not embedding or not hasattr(embedding, 'data') or not embedding.data:
           raise ValueError("Received invalid embedding response from OpenAI")
           
       embedding_vector = embedding.data[0].embedding
       vector_length = len(embedding_vector)
       
       if vector_length == 0:
           raise ValueError("Received empty embedding vector from OpenAI")
           
       print(f"Dimension of query embedding: {vector_length}")
       return embedding
   except (AttributeError, IndexError) as struct_err:
       error_msg = f"Malformed response from OpenAI embedding API: {str(struct_err)}"
       print(error_msg)
       raise ValueError(error_msg) from struct_err
   except Exception as e:
       error_msg = f"Error generating embeddings: {str(e)}"
       print(error_msg)
       raise RuntimeError(error_msg) from e

def create_embeddings(text, model_emb):   
    response = openai_client.embeddings.create(
        input=text,
        model=model_emb
    )
    embedding = response.data[0].embedding
    return embedding     

# create prompt for openai
def create_prompt(query, res):
    contexts = [ x['metadata']['text'] for x in res['matches']]
    prompt_start = ("Answer the question based on the context and sentiment of the question.\n\n" + "Context:\n") # also, do not discuss any Personally Identifiable Information.
    prompt_end = (f"\n\nQuestion: {query}\nAnswer:")
    prompt = (prompt_start + "\n\n---\n\n".join(contexts) + prompt_end)
    return prompt


def add_prompt_messages(role, content, messages):
    json_message = {
        "role": role, 
        "content": content
    }
    messages.append(json_message)
    return messages

def get_chat_completion_messages(messages, model_chat, temperature=0.0): 
    try:
        if not messages:
            raise ValueError("No messages provided for chat completion")
            
        if not model_chat:
            raise ValueError("No chat model specified")
            
        # Validate messages format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format: {msg}")
                
        # Set reasonable timeout and max retries
        response = openai_client.chat.completions.create(
            model=model_chat,
            messages=messages,
            temperature=temperature,
            timeout=30,  # 30 second timeout
        )
        
        if not response or not hasattr(response, 'choices') or not response.choices:
            raise ValueError("Received invalid completion response from OpenAI")
            
        if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
            raise ValueError("No message in completion response")
            
        if not hasattr(response.choices[0].message, 'content'):
            raise ValueError("No content in completion message")
            
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as struct_err:
        error_msg = f"Malformed response from OpenAI completion API: {str(struct_err)}"
        print(error_msg)
        raise ValueError(error_msg) from struct_err
    except Exception as e:
        error_msg = f"Error in chat completion: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def create_system_prompt():
    system_prompt = f"""
    You are a customer service specialist at a multiple listing service that helps customers.
    """
    return system_prompt

