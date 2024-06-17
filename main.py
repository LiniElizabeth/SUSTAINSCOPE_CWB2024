import streamlit as st
import seaborn as sns
import os
from textblob import TextBlob
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAI,ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser,CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from langchain_community.vectorstores.azuresearch import AzureSearch
from DataRetrievalModules import read_summaries_from_file,generate_insights,write_summaries_to_file,retrieve_docs_similarity
from langchain.chains.question_answering import load_qa_chain

#Loading Environment Variables
load_dotenv()

#Obtaining Azure AI Search Credentials 
AZURE_AI_SEARCH_ENDPOINT=os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY=os.getenv("AZURE_AI_SEARCH_API_KEY")  
AZURE_AI_SEARCH_INDEX_NAME=os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
#search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, index_name=AZURE_AI_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY))

#Obtaining OpenAI Credentials 
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#Initialise OpenAI Model
llm = OpenAI(temperature=0, max_tokens=500, model = "gpt-4o")
model=ChatOpenAI(temperature=0, max_tokens=500, model = "gpt-4o")
model_chat=ChatOpenAI(temperature=0.2, max_tokens=500, model = "gpt-4o")
#Initialise OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Streamlit Design
icon_path = "logo.png"
st.set_page_config(page_title="SUSTAINSCOPE", page_icon=icon_path,
                            layout='centered', initial_sidebar_state="collapsed")
_,_,logo,_,_ = st.columns(5)
logo.image(icon_path, width=150)
style = ("text-align:center; padding: 0px; font-family: arial black;, "
            "font-size: 250%; color: blue;")
title = f"<h1 style='{style}'>SUSTAINSCOPE</h1><br><br>"
st.write(title, unsafe_allow_html=True)

# Disclaimer at the bottom
st.markdown(
            """
            <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f8f9fa;
                color: black;
                text-align: center;
                padding: 10px;
                border-top: 0px solid #e9ecef;
            }
            </style>
            <div class="footer">
                <p style='font-style: italic;'>
                Disclaimer: This application uses AI-generated content for insights. While we strive for accuracy, please verify findings with primary sources.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Centered selection boxes
company = st.selectbox("Select the Company for Analysis", ["Choose an option","RIO TINTO", "BHP"], key='company')
year = st.selectbox("Select Financial Year", ["Choose an option","2023", "2022"], key='year')
action=st.selectbox("Choose the feature options", ["Choose an option","Generate Sustainability Progress", "Chat with Sustainability Documents","View Public perception"], key='action')





def user_input(user_query,filter):

    company_report_docs=retrieve_docs_similarity(user_query,filter)
    company_report_content = " ".join([doc["content"] for doc in company_report_docs])
    prompt_template = """
                        You are a sustainability analyst with great expertise.Answer the question {user_query}from the 
                        provided context{company_report_content} only, make sure to provide all the details,
                        don't provide the wrong answer.Answer to the point\n\n
                    
                    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages=prompt.format_messages(company_report_content=company_report_content,user_query=user_query)
    chat_response=model_chat.invoke(messages)
    st.write("Reply: ", chat_response.content)

@st.cache_data
def print_sentiment_and_wordcloud(news_summary_list):
    # Extract summaries and sentiment 

    summaries = [item['summary'] for item in news_summary_list]
    sentiment = " ".join([item['sentiment analysis'] for item in news_summary_list])

    # Plot sentiment scores
    st.write("Sentiment Analysis of News :\n",sentiment)

    # Generate word cloud
    text = ' '.join(summaries)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt) 
    plt.close()   

@st.cache_data
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
    SentimentResponseSchema=ResponseSchema(name="sentiment analysis",description="Discuss the sentiment of the article with reasoning.")
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
        return news_summary_list


@st.cache_data
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
    # Create the chains using runnables
    section_response_final=model.invoke(messages)

    return section_response_final.content

# Streamlit app


news_summary_list=generate_news_summary()

def click_button():
    if company != "Choose an option" and year != "Choose an option" and action == "Generate Sustainability Progress":
        
        report_section_env=generate_section_reports(company=company,year=year,pillar="Environmental")
        report_section_soc=generate_section_reports(company=company,year=year,pillar="Social")
        report_section_gov=generate_section_reports(company=company,year=year,pillar="Governance")  
        st.markdown("""
                <style>
                    .report-container {
                        border: 2px solid #e0e0e0;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    .report-header {
                        font-size: 24px;
                        font-weight: bold;
                        text-align: center;
                        margin-bottom: 20px;
                    }
                    .report-subheader {
                        font-size: 20px;
                        font-weight: bold;
                        color: #4CAF50;
                        margin-top: 20px;
                        margin-bottom: 10px;
                    }
                    .report-content {
                        font-size: 16px;
                        line-height: 1.6;
                    }
                </style>
                """, unsafe_allow_html=True)

        st.markdown(f"<div class='report-header'>SUSTAINABILITY PROGRESS REPORT</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container'><strong>COMPANY:</strong> {company}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container'><strong>Financial Year:</strong> {year}</div>", unsafe_allow_html=True)

        st.markdown("<div class='report-subheader'>Environmental Sustainability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container report-content'>{report_section_env}</div>", unsafe_allow_html=True)

        st.markdown("<div class='report-subheader'>Social Sustainability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container report-content'>{report_section_soc}</div>", unsafe_allow_html=True)

        st.markdown("<div class='report-subheader'>Governance Sustainability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-container report-content'>{report_section_gov}</div>", unsafe_allow_html=True)    

    elif company != "Choose an option" and year != "Choose an option" and action == "Chat with Sustainability Documents":
        user_query = st.text_input("Ask a Question about company's sustainability reports")
        if user_query:
            filter=f"company_name eq '{company}' and document_type eq 'COMPANY REPORTS' and financial_year eq '{year}'"
            user_input(user_query,filter)

    elif company != "Choose an option" and year != "Choose an option" and action == "View Public perception":
        print_sentiment_and_wordcloud(news_summary_list)

    
        

click_button()





        
    




