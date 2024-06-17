#DataRetrievalModules.py

import streamlit as st
from textblob import TextBlob
import openai
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_openai import OpenAI,ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser,CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.runnables import RunnablePassthrough

#Loading Environment Variables
load_dotenv()

#Obtaining Azure AI Search Credentials 
AZURE_AI_SEARCH_ENDPOINT=os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY=os.getenv("AZURE_AI_SEARCH_API_KEY")  
AZURE_AI_SEARCH_INDEX_NAME=os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, index_name=AZURE_AI_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))

#Obtaining OpenAI Credentials 
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#Initialise OpenAI Model
#llm = OpenAI(temperature=0, max_tokens=500, model = "gpt-4o")
model=ChatOpenAI(temperature=0, max_tokens=2048, model = "gpt-4o")
#Initialise OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


#Function to write summaries to a text file
def write_summaries_to_file(pillar_name, summaries):
    filename = f"{pillar_name}_summaries.txt"
    with open(filename, "w") as file:
        for summary in summaries:
            file.write(summary + "\n\n")  # Write each summary followed by a blank line for separation

# Function to read summaries from a file and store in a variable
def read_summaries_from_file(pillar_name):
    filename = f"{pillar_name}_summaries.txt"
    with open(filename, "r") as file:
        content = file.read()
    return content


#Function to generate news summaries, obtain ESG label and perform sentiment analysis on news articles
def generate_news_summary():
    
    search_client = SearchClient(AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))

    results = search_client.search(
            search_text="",
            select=["content"],
            filter=f"document_type eq 'NEWS MEDIA'"
    )

    #Defining the context for LLM to analyse the news articles
    context='''Environmental, social, and governance (ESG) assesses business practices on sustainability and ethics.
            It measures risks and opportunities related to:
            - Environmental: Emissions (GHG, pollution), resource use (virgin vs. recycled materials, cradle-to-grave cycling), water stewardship, land use (deforestation, biodiversity).
            - Social: Employee development, labor practices, product safety, supply chain standards, access for underprivileged groups.
            - Governance: Shareholder rights, board diversity, executive compensation alignment with sustainability, corporate behavior (anti-competitive practices, corruption).

            '''  

    SummaryResponseSchema=ResponseSchema(name="summary", description="Summarise key ponts given in text from the point of view of sustainability")
    SentimentResponseSchema=ResponseSchema(name="sentiment analysis",description="Analyse the sentiment with reasoning")
    PillarResponseSchema=ResponseSchema(name="pillar",description="Assign Environmental,Social or Governance")             
        
    response_schemas=[SummaryResponseSchema,SentimentResponseSchema,PillarResponseSchema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()


    news_summary_list=[]
    #Accessing the news aricles one by one
    for result in results:
        
        
        article=result["content"]
        # Define the prompt templates
        prompt_template = """
                            You are a sustainability analyst with expert knowledge of the following context: {context}.Analyze the following news article: {article}.
                            summary:Generate a summary mentioning all the key points related to the company's sustainability efforts.
                            pillar:Specify sustainability pillar one of(Environmental, Social, or Governance) 
                            sentiment_analysis:Discuss the sentiment of the article with reasoning.
                            Do not include any content that is not present or derived from the article. Do not include any comments
                            Always put a delimiter ',' between the three key-value pairs
                            Format the response according to the following instructions: 
                            {format_instructions}
                        """         

        articles_prompt = ChatPromptTemplate.from_template(prompt_template)
        messages=articles_prompt.format_messages(context=context,article=article ,format_instructions=format_instructions)
        # Create the chains using runnables
        response_final=model.invoke(messages)
        response_final_dict=output_parser.parse(response_final.content)
        response_final_dict["sentiment_score"]=TextBlob(article).sentiment.polarity
        news_summary_list.append(response_final_dict)

        # Filter and write summaries for each pillar
        for pillar in ["Environmental", "Social", "Governance"]:
            # Extract summaries for the current pillar
            summaries = [news["summary"] for news in news_summary_list if news["pillar"] == pillar]
            # Write these summaries to a file
            write_summaries_to_file(pillar, summaries)


#Function to generate queries from summaries in order to compare and verify sustainability efforts
def generate_insights(summaries):
    context='''Environmental, social, and governance (ESG) assesses business practices on sustainability and ethics.
            It measures risks and opportunities related to:
            - Environmental: Emissions (GHG, pollution), resource use (virgin vs. recycled materials, cradle-to-grave cycling), water stewardship, land use (deforestation, biodiversity).
            - Social: Employee development, labor practices, product safety, supply chain standards, access for underprivileged groups.
            - Governance: Shareholder rights, board diversity, executive compensation alignment with sustainability, corporate behavior (anti-competitive practices, corruption).

            '''  

    FirstQueryResponseSchema=ResponseSchema(name="Q1",description="First Query")     
    SecondQueryResponseSchema=ResponseSchema(name="Q2",description="Second Query")  
    ThirdQueryResponseSchema=ResponseSchema(name="Q3",description="Third Query")  
    FourthQueryResponseSchema=ResponseSchema(name="Q4",description="Fourth Query")  
    FifthQueryResponseSchema=ResponseSchema(name="Q5",description="Fifth Query")  
    response_schemas=[FirstQueryResponseSchema,SecondQueryResponseSchema,ThirdQueryResponseSchema,FourthQueryResponseSchema,FifthQueryResponseSchema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)  
    format_instructions = output_parser.get_format_instructions()
    # Define the prompt template
    prompt_template = """
                        You are a sustainability analyst with expert knowledge of the following context: {context}.
                        You are investigating if a company is actually working towards its sustainability targets and goals
                        First Analyze the following news summary: {summaries}. Then generate 5 queries[Q1,Q2,Q3,Q4,Q5] based only on
                        what is mentioned in the given news summary, answers to which should be present in the company reports such as 
                        sustainability report,annual report ,investor presentation report and climate report. The queries should be such
                        that their answers should help you compare the company's actual sustainability efforts reported in the media and 
                        the (goals, targets ,key performance indicators) claimed by the company in its report. Think critically and ask the 
                        correct queries to unearth comparison points from the company reports which will later help us generate a company 
                        sustainability progress report.In the query, if you refer to any content from the news summary,explicitly add the 
                        information to your query instead of stating as mentioned in news summary. Make sure the 5 queries help to think 
                        critically about all the points mentioned in the summaries.
                        {format_instructions}
                    """         

    query_generator_prompt = ChatPromptTemplate.from_template(prompt_template)
    messages=query_generator_prompt.format_messages(context=context,summaries=summaries ,format_instructions=format_instructions)
    # Create the chains using runnables
    query_response_final=model.invoke(messages)
    query_response_final_dict=output_parser.parse(query_response_final.content)
    return query_response_final_dict

#Function to perform Vector search in Azure AI Search to rtrieve documents based on query
def retrieve_docs_similarity(query,filter):
    #Retrieves answers from the Company Reports stored in Vector Store for a given query
    search_client = SearchClient(AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))
    vector_query = VectorizedQuery(vector=embeddings.embed_query(query), k_nearest_neighbors=3, fields="content_vector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        filter=filter
    )

    return (results)

#Generate sustainability progress reports sections
def generate_section_reports(company,year,pillar):

    # Reading summaries for given pillar and storing them in variables
    summaries = read_summaries_from_file(pillar)
    
    #Generate queries corresponding to points in the news media
    queries=generate_insights(summaries)
    

    
    filter=f"company_name eq '{company}' and document_type eq 'COMPANY REPORTS' and financial_year eq '{year}'"
    company_report_data_list=[]
    query_list=[]
    for query in queries:
        query_list.append(query)
        company_report_docs=retrieve_docs_similarity(query,filter)
        company_report_data_list.extend(company_report_docs) #Flattening the list
    company_report_content = " ".join([doc["content"] for doc in company_report_data_list])


    context='''Environmental, social, and governance (ESG) assesses business practices on sustainability and ethics.
            It measures risks and opportunities related to:
            - Environmental: Emissions (GHG, pollution), resource use (virgin vs. recycled materials, cradle-to-grave cycling), water stewardship, land use (deforestation, biodiversity).
            - Social: Employee development, labor practices, product safety, supply chain standards, access for underprivileged groups.
            - Governance: Shareholder rights, board diversity, executive compensation alignment with sustainability, corporate behavior (anti-competitive practices, corruption).

            '''  

    # Define the prompt template
    prompt_template = """
                        You are a sustainability analyst with expert knowledge of the following context: {context}.
                        You are investigating if a company is actually working towards its sustainability targets and goals.
                        Task:
                        1)Review the following news summaries: {summaries} and the queries {queries}that arise from them. 
                        Critically verify if the news data which represents the company's sustainability
                        efforts published in the media aligns with the targets and goals claimed by the company in its reports.
                        2)Analyse the information {company_report_content}retrieved from the company reports.Use this information 
                        to answer the queries arising from the news summaries. 
                        
                        Report Structure:
                        Generate a section of the report having the following schema:

                        1)**News Media Observations**:- Mention all the relevant key points from the {summaries} in bulleted points.
                                                     - Highlight the points you want to compare with the sustainability company report data.

                        2)**Company Report Data**:- Compare the key points mentioned in the News Media Observations with the claims,
                                                    targets,goals,objectives, key performance indicators retrieved from {company_report_content}.

                        3)**Actions & Insights**: - Based on the above two sections, use your skills as a sustainability analyst to critically reason and provide actions and insights.
                                                  - Offer recommendations for the company to improve their efforts towards the {pillar} sustainability pillar.
                                                  - Ensure the actions and insights are valuable to investors, NGOs, sustainability teams, and public policy makers.

                        4)**Sentiment Analysis**:- Discuss the public perception of {company}'s {pillar} sustainability pillar based on sentiment analysis of the news summaries.
                                                - Provide detailed reasoning for your analysis.

                        Important Notes:
                        - All content generated should be based only on the {context}, {summaries}, and {company_report_content}.
                        - Ensure that you generate content for all four schema headers (News Media Observations, Company Report Data, Actions & Insights, Sentiment Analysis).
                        - Exclude any information not related to {company}'s sustainability efforts.
                        -generated content should include info only corresponding to the sustainability pillar{pillar}

                        
                    """         

    section_generator_prompt = ChatPromptTemplate.from_template(prompt_template)
    messages=section_generator_prompt.format_messages(context=context,summaries=summaries,company_report_content=company_report_content,queries=queries ,pillar=pillar,company=company)
    # Invoking Model
    section_response_final=model.invoke(messages)

    return section_response_final.content


#Performing generation of news summaries, obtain ESG label and perform sentiment analysis on news articles
#and writing to a file
generate_news_summary()