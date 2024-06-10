#Importing Required Libraries
import os
import requests
import json
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader,NewsURLLoader,PyPDFLoader,UnstructuredExcelLoader
from newspaper import Article
from datetime import datetime
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential 
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchFieldDataType,
    HnswParameters
)

#Loading Environment Variables
load_dotenv()

#Obtaining Azure Blob Storage Credentials
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_CONTAINER_NAME=os.getenv('AZURE_STORAGE_CONTAINER_NAME')

#Obtaining Azure AI Search Credentials 
AZURE_AI_SEARCH_ENDPOINT=os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY=os.getenv("AZURE_AI_SEARCH_API_KEY")  

#Obtaining OpenAI Credentials 
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#Accessing Data from Azure Blob Storage & Data Preprocessing

# Ensure the download folder exists
download_folder = 'data_sources'
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

# List blobs in the container
blob_list = container_client.list_blobs()

# Download and process each file
documents=[]
for blob in blob_list:
    blob_name = blob.name
    blob_client = blob_service_client.get_blob_client(container=AZURE_STORAGE_CONTAINER_NAME, blob=blob_name)
    print(f"Processing blob: {blob_name}")
    download_path = os.path.join(download_folder, blob_name)
    
    # Download the blob
    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    
    # Process PDF files
    if blob_name.endswith('.pdf'):
        # Fetch the metadata
        blob_metadata = blob_client.get_blob_properties().metadata
        loader = PyPDFLoader(download_path)
        data = loader.load()
        for doc in data:
            # update metadata for each document
            doc.metadata.update(blob_metadata)  # update metadata for each document
            documents.append(doc)
       
    # Process Excel files
    elif blob_name.endswith('.xls') or blob_name.endswith('.xlsx'):
         # Fetch the metadata
        blob_metadata = blob_client.get_blob_properties().metadata
        df = pd.read_excel(download_path)
        for index in range(len(df)):
            url = [df['URL'].iloc[index]]
            loader = NewsURLLoader(url)
            data=loader.load()
            for doc in data:
                # update metadata for each document
                doc.metadata.update(blob_metadata)
                metadata_additional= { "company_name": df['COMPANY_NAME'].iloc[index]} 
                doc.metadata.update(metadata_additional) 
                documents.append(doc)

#Convert datetime objects in metadata to string format for Azure AI Search index compatibility
for doc in documents:
    for key, value in doc.metadata.items():
        if isinstance(value, datetime):
            doc.metadata[key] = value.isoformat()

# Define the fields for the index
index_name = "cwb-2024-ss-index"
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="document_name", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True),
    SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
    SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="company_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="financial_year", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="publish_date", type=SearchFieldDataType.DateTimeOffset, filterable=False, facetable=False),
    SearchField(
        name="content_vector", 
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True, 
        vector_search_dimensions=1536, 
        vector_search_profile_name="myHnswProfile")
    ]

# Configure the vector search configuration  
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw"
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        )
    ]
    

)

#Creating Search index with vector search configuration
index = SearchIndex(name=index_name, fields=fields,vector_search=vector_search)

# Initialize the SearchIndexClient
index_client = SearchIndexClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))

# Create the index
# Check if index exists and delete if it does
try:
    index_client.get_index(index_name)
    index_client.delete_index(index_name)
except:
    pass  # Index doesn't exist yet, no need to delete

result=index_client.create_index(index)
print(f'{result.name} created')

#Creating Chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

#Initialise OpenAI Model
llm = OpenAI(temperature=0.9, max_tokens=500, model = "gpt-3.5-turbo-instruct")
#Initialise OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define a function to create embeddings for a batch of texts
def create_embeddings_batch(texts, model):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "input": texts,
        "model": model
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    embeddings_list = [item['embedding'] for item in result['data']]
    return embeddings_list

# Function to process documents in batches
def process_documents_in_batches(docs, batch_size, model):
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_embeddings = create_embeddings_batch(batch_texts, model)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

# Define batch size
batch_size = 16  # Adjust based on your needs and API rate limits

# Process documents and generate embeddings
embeddings = process_documents_in_batches(chunks, batch_size, "text-embedding-3-small")


# Transform data into JSON
docs = []
for index, (doc, embedding) in enumerate(zip(chunks, embeddings)):
    docs.append({
        "id": str(index + 1),
        "document_name": doc.metadata.get("document_name"),
        "content": doc.page_content,
        "document_type": doc.metadata.get("document_type"),
        "company_name": doc.metadata.get("company_name"),
        "financial_year": doc.metadata.get("financial_year"),
        "publish_date": doc.metadata.get("publish_date"),
        "content_vector": embedding
    })

json_data = json.dumps(docs, indent=2)

with open('SustainScopeData.json', 'w') as f:
    f.write(json_data)

 # Convert date fields to the correct format
def format_date(date_str):
    try:
        # Parse the date string
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        # Convert to the required format with UTC timezone
        return dt.isoformat() + 'Z'
    except ValueError:
        # If the date format is incorrect or None, return None
        return None

# Load data from JSON file and format dates
with open('SustainScopeData.json', 'r') as f:
    documents = json.load(f)
    for doc in documents:
        if 'publish_date' in doc and doc['publish_date']:
            doc['publish_date'] = format_date(doc['publish_date'])
        if 'financial_year' in doc and doc['financial_year']:
            doc['financial_year'] = format_date(doc['financial_year'])

# Initialize the SearchClient with AzureKeyCredential
search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))

# Upload documents
result = search_client.upload_documents(documents)
print(f"Uploaded {len(documents)} documents in total")


