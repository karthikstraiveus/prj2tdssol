from typing import Union

from fastapi import FastAPI,Request, File, UploadFile, Form

from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import SimilarityFunction
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from urllib.request import urlopen
import markdownify 
from datetime import datetime
import re
from urllib.parse import urlencode
import camelot
import pandas as pd

import requests
import base64
from PIL import Image

import numpy as np

import json
import os
import io
from typing import Dict, Any, Annotated

from itertools import islice

#from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from math import pi
from typing import Union
from pypdf import PdfReader
import time

from pydantic import Extra
import requests
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from zipfile import ZipFile
from langchain_core.tools import tool
from langchain.agents.self_ask_with_search.output_parser import SelfAskOutputParser

class LlamaLLM(LLM):
    llm_url: str = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "gpt-4o-mini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        #if stop is not None:
            #raise ValueError("stop kwargs are not permitted.")
        #sys_msg = ''' You are helpful Assistant if you are unable to answer use tools and run it '''
        sys_msg = ''' You are helpful Assistant if you are unable to answer use tools '''

        payload = {
           "model": "gpt-4o-mini",  "temperature":0, "messages": [{"role": "system","content": sys_msg},{"role": "user", "content": prompt}],
        }

        response = requests.post(self.llm_url, json=payload, headers={"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}"}, verify=False)
        response.raise_for_status()

        #print("API Response:", response.json())

        output_response = response.json()

        output_details = output_response['choices'][0]['message']['content']

        return output_details
        

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}


class Item(BaseModel):
    question: str

app = FastAPI(debug=True)

app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"],  allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"]) # Allow GET requests from all origins


class CircumferenceTool(BaseTool):
    name: str = "Circumference calculator"
    description: str = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")


class DownloadunzipTool(BaseTool):
    name: str = "Download and unzip file"
    description: str = "use this tool when you need to download and unzip file"

    def _run(self, filename: str):
        file_location = f"files/{filename}"
        file_name, file_extension = os.path.splitext(file_location)
        #print(file_name)
        #print(file_extension)
        with ZipFile('files/'+filename, "r") as zip:
            zip.extractall(path=file_name)

        for file in os.listdir(file_name): 
            file_path = file_name+"/"+file
            with open(file_path, 'r') as f: 
                content = f.read()
        return content
        

    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")


class FindHiddeninput(BaseTool):
    name: str = "Hidden input value"
    description: str = "use this tool when you need to find the hidden input in html"

    def _run(self, filename: str):
        filename_title = filename["title"]

        file_location = f"files/{filename_title}"
        with open(file_location, 'r') as f: 
                content = f.read()
        return content

    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")

class GitFileupload(BaseTool):
    name: str = "commit file to github repo"
    description: str = "use this tool when you need to commit file to github repository"

    def _run(self, filename: str):

        repo_owner = "karthikstraiveus"
        repo_name = "gitpagestest"
        file_path = filename

        if "title" in filename:
            filename_title = filename["title"]
        else:
            filename_title = filename

        file_location = f"files/{filename_title}"
        with open(file_location, 'rb') as f: 
                content = f.read()

        file_contents = content

        # Replace with your GitHub personal access token
        access_token = ""

        # Set the API endpoint and headers
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Set the file contents and commit message
        data = {
            "message": f"Create {file_path}",
            "content": base64.b64encode(file_contents).decode('utf-8')
        }

        # Convert the data to JSON
        json_data = json.dumps(data)

        # Send the POST request to create the file
        response = requests.put(url, headers=headers, data=json_data)

        # Check if the file was created successfully
        repo_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/email.json"
        if response.status_code == 201:
            print(f"File {file_path} created successfully!")
            return repo_url
        else:
            print(f"Error creating file: {response.text}")
            return "Error in commit file in to the github"
    

    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")


class ImageCompression(BaseTool):
    name: str = "Download the image and compress it losslessly"
    description: str = "use this tool when you need to download the image and compress it losslessly"

    def _run(self, filename: str):

        if "title" in filename:
            filename_title = filename["title"]
        else:
            filename_title = filename

        input_image_path = f"files/{filename_title}"

        with Image.open(input_image_path) as img:
            # Convert the image to RGB if it's not in that mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Initialize variables for compression
            quality = 100  # Start with the highest quality
            buffer = io.BytesIO()
            target_size = 1500
            
            # Compress the image in a loop until the size is less than target_size
            while True:
                # Save the image to the buffer in WebP format
                img.save(buffer, format='WEBP', quality=quality)
                size = buffer.tell()  # Get the size of the image in bytes
                
                if size <= target_size:
                    break  # Exit the loop if the size is acceptable
                
                # Decrease quality for the next iteration
                quality -= 5  # Decrease quality by 5 for the next iteration
                
                # Reset the buffer for the next save
                buffer.seek(0)
                buffer.truncate()
                
                if quality < 0:
                    raise ValueError("Unable to compress image to the desired size.")
            
            # Get the byte data from the buffer
            buffer.seek(0)
            image_data = buffer.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return base64_image

        
    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")


class PDFMarkDown(BaseTool):
    name: str = "convert the pdf to markdown"
    description: str = "use this tool when you need to convert the pdf to markdown or get the markdown content from pdf"

    def _run(self, filename: str):

        if "title" in filename:
            filename_title = filename["title"]
        else:
            filename_title = filename

        file_location = f"files/{filename_title}"
        reader = PdfReader(file_location)

        output_text = ""

        for page in reader.pages:
            text = page.extract_text()
            output_text += text
        
        return output_text

    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")   


class VercelDeploy(BaseTool):
    name: str = "deploy a Python app to Vercel"
    description: str = "use this tool when you need to deploy a python app to vercel"

    def _run(self, filename: str):

        # Set your Vercel token and project details
        VERCEL_TOKEN = 'your_vercel_token'
        VERCEL_PROJECT_NAME = 'your_project_name'
        VERCEL_TEAM_ID = 'your_team_id'  # Optional, if you're using a team

        # Path to your FastAPI app directory
        APP_DIR = 'path/to/your/my-fastapi-app'

        # Change to the app directory
        os.chdir(APP_DIR)

        # Prepare the deployment
        headers = {
            'Authorization': f'Bearer {VERCEL_TOKEN}',
            'Content-Type': 'application/json'
        }

        # Create a deployment
        deployment_data = {
            "name": VERCEL_PROJECT_NAME,
            "project": VERCEL_PROJECT_NAME,
            "teamId": VERCEL_TEAM_ID,  # Optional
            "files": [
                {
                    "file": "api.py",
                    "data": open("api.py", "r").read()
                },
                {
                    "file": "requirements.txt",
                    "data": open("requirements.txt", "r").read()
                },
                {
                    "file": "vercel.json",
                    "data": open("vercel.json", "r").read()
                }
            ]
        }

        response = requests.post('https://api.vercel.com/v1/deployments', headers=headers, json=deployment_data)

        if response.status_code == 201:
            print("Deployment successful!")
            print("Deployment URL:", response.json().get('url'))
        else:
            print("Deployment failed!")
            print("Response:", response.json())

        
    def _arun(self, filename: str):
        raise NotImplementedError("This tool does not support async")
    


class WeatherForecast(BaseTool):
    name: str = "weather forecast for city"
    description: str = "use this tool when you need to weather forecast a city"

    def _run(self, cityname: str):

        output_format = {}
        required_city = "Brussels"
        location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': '',
        's': required_city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
        })

        result = requests.get(location_url,verify=False).json()
        #print(result)

        weather_brokerurl = 'https://www.bbc.com/weather/'+result['response']['results']['results'][0]['id']
        response = requests.get(weather_brokerurl,verify=False)

        soup = BeautifulSoup(response.content,'html.parser')

        forecast_date_values = soup.find_all('div', attrs={'class': 'wr-day__title wr-js-day-content-title'})
        forecast_summary_values = soup.find_all('div', attrs={'class': 'wr-day__weather-type-description wr-js-day-content-weather-type-description wr-day__content__weather-type-description--opaque'})


        for summary in forecast_summary_values:
            for date_value in forecast_date_values:
            
                date_string = date_value['aria-label'] + ' 2025'
                date_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_string)
                d = datetime.strptime(date_string, "%A %d %B %Y")
                ymd_format = d.strftime("%Y-%m-%d")
                output_format[ymd_format] = ''
            
                
                summary_text = summary.get_text()
                output_format[ymd_format] = summary.get_text()
        return output_format


    def _arun(self, cityname: str):
        raise NotImplementedError("This tool does not support async")

class CountryWikipedia(BaseTool):
    name: str = "fetch wikipedia content of the country"
    description: str = "use this tool when you need to fetch wikipedia content of the country"

    def _run(self, country: str):
        
        markdown_html = ''
        wiki_url = "https://en.wikipedia.org/wiki/"+country

        html = urlopen(wiki_url).read().decode("utf-8")
        # using beautifulsoup
        bs = BeautifulSoup(html, 'html.parser')
        heading_list = bs.find_all(['h1', 'h2', 'h3','h4','h5','h6'])
        #print(heading_list)
        for headings in heading_list:
            html_heading = f'{headings}'
            markdown_html += markdownify.markdownify(html_heading, heading_style="ATX")
            
        #markdown_html = markdownify.markdownify(heading_list, heading_style="ATX")
        return markdown_html
    
    def _arun(self, country: str):
        raise NotImplementedError("This tool does not support async")
    
    
class LocateGitUsers(BaseTool):
    name: str = "locate users in city and followers using github api"
    description: str = "use this tool when you need to locate users in city and followers using github api"

    def _run(self, city: str, followers: int):

        # Replace 'your_token' with your actual GitHub personal access token
        headers = {
            'Authorization': '',
        }

        # Search for users in Moscow
        url = 'https://api.github.com/search/users?q=location:'+city
        response = requests.get(url, headers=headers, verify=False)

        print(response.status_code)
        if response.status_code == 200:
            users = response.json().get('items', [])
            
            # Filter users with more than 170 followers
            filtered_users = []
            created_at_list = []

            for user in users:
                user_url = user['url']
                user_response = requests.get(user_url, headers=headers, verify=False)
                time.sleep(3)
                #print(user_response.content)
                
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    if user_data.get('followers', 0) > followers:
                        filtered_users.append(user_data)
            
            # Print the filtered users

            #print("filtered users")
            print(filtered_users)
            for user in filtered_users:
                print(f"Username: {user['login']}, Followers: {user['followers']}, Profile URL: {user['html_url']}, Created At: {user['created_at']}")

                #get_created_at = requests.get(user['url'])

                #print(get_created_at.content)

                #if get_created_at.status_code == 200:
                    #created_at_data = get_created_at.json()
                    #created_at_list = created_at_data.get('created_at')
            
            #print(created_at_list)
        else:
            print(f"Error fetching users: {response.status_code}")
    
    def _arun(self,  city: str, followers: int):
        raise NotImplementedError("This tool does not support async")


class ExtractTablePDF(BaseTool):
    name: str = "extract data from pdf and calculate the marks for the subjects"
    description: str = "use this tool when you need to extract data from pdf and calculate the marks for the subjects"

    def _run(self, filename: str, totalsubject: str, scoremarks: str, subject: str, groups: str):

        if "title" in filename:
            filename_title = filename["title"]
        else:
            filename_title = filename

        if "title" in groups:
            groups = groups["title"]
        else:
            groups = groups

        file_location = f"files/{filename_title}"
        tables = camelot.read_pdf(file_location,pages=groups)
        dataframe_list = []

        split_groups = groups.split("-")
        range_from  = (int(split_groups[1])- int(split_groups[0])) + 1

        x = range(0, range_from)
        for n in x:
            print(tables[n])
            tables[n].df
            dataframe_list.append(tables[n].df)

        result = pd.concat(dataframe_list)
        result.drop(index=result.index[0], axis=0, inplace=True)
        #print(result)


        #print(result[0])
        result[0] = pd.to_numeric(result[0], errors='coerce')
        result[1] = pd.to_numeric(result[1], errors='coerce')
        result[2] = pd.to_numeric(result[2], errors='coerce')
        result[3] = pd.to_numeric(result[3], errors='coerce')
        result[4] = pd.to_numeric(result[4], errors='coerce')

        if "title" in scoremarks:
            scoremarks = scoremarks["title"]
        else:
            scoremarks = scoremarks

        result=result[result[4] >= int(scoremarks)]

        sum_economics = result[0].sum()

        return str(sum_economics)

    def _arun(self, filename: str, totalsubject: str, scoremarks: int, subject: str, groups: str):
        raise NotImplementedError("This tool does not support async")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/")
#def get_question_match(item: Item, file: Annotated[bytes, File()]):
def get_question_match(question: Annotated[str, Form()], file: UploadFile = File(None)):

    #req_q = item.question
    #question = question + '''{ file }'''
    #print(question)

    if file:

        print("the length of file is " + file.filename)
        file_location = f"files/{file.filename}"

        # Check if the file exists
        if os.path.exists(file_location):
            os.remove(file_location)
            
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        #getfile_details = os.path.splitext(file_location)
        getfile_details = os.path.splitext(file.filename)
        print(getfile_details)
        question = question + ''' the filename is ''' + getfile_details[0]+getfile_details[1]

        
        if not os.path.exists(getfile_details[0]):
            os.umask(0)
            os.makedirs(getfile_details[0])
            
        #os.chmod(file_name,0o777) 

        #if file_extension == ".zip":
            #with ZipFile('files/'+file.filename, "r") as zip:
                #zip.extractall(path=file_name)

            #for file in os.listdir(file_name): 
                #file_path = file_name+"/"+file
                #with open(file_path, 'r') as f: 
                    #content = f.read()

            #question = question + content
    
    question = question + ''' give the final answer only in json format with answer as key '''
    print(question)
    
    llm = LlamaLLM()
    tools = [DownloadunzipTool(),FindHiddeninput(),GitFileupload(),ImageCompression(),PDFMarkDown(),WeatherForecast(),CountryWikipedia(),
             LocateGitUsers(), ExtractTablePDF()]

    # initialize conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
        )

    agent = initialize_agent(
    agent='structured-chat-zero-shot-react-description',
    tools=tools,
    llm=llm,
    verbose=False,
    max_iterations=1,
    early_stopping_method='generate',
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    memory=conversational_memory
    )

    #response = agent.run(question)
    response = agent.invoke(question)
    return json.loads(response["output"])
