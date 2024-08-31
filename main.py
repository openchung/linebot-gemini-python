from linebot.models import FlexSendMessage
from linebot.models import (
    MessageEvent, TextSendMessage
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from fastapi import Request, FastAPI, HTTPException
import google.generativeai as genai

# langchain imports
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import sys
from io import BytesIO

import aiohttp
import PIL.Image
import pytesseract
import requests
import json

from dotenv import load_dotenv

load_dotenv()
# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
gemini_key = os.getenv('GEMINI_API_KEY')

images_prompt = '''
You are playing an AI assistant with strong image analysis capabilities, as well as a math and programming expert.
When you receive a picture, please classify the picture first to determine what the user may want to ask you, which of the following questions it is:
1. Information type (geography, physics, scientific questions)
Interpret information content and summarize it, translate it into Chinese and explain it with scientific details

2. Mathematics or programming problems
- Read the question content first to confirm whether there are input, output and explanation examples. If so, generate the verification method and verification program before starting to solve the problem.
- The problem-solving process must briefly explain what kind of formula is applied. The most important thing is to use Java and Python programming languages ​​to generate two program code functions respectively, and they must comply with the functions that can be called by the main method. Before generating, Make inferences and verify whether the answers you produce are correct. After output, use the previous verification program to verify whether the output function is correct and confirm again. If correct, output the answer; if incorrect, restart the previous process.
- If the program cannot be produced or the problem cannot be solved, directly output "I cannot handle this math or program"

3. Chinese language or history issues
Please find the answer based on the history or literary creation of each country, and discuss the source.

4. Character issues
Identify the origin of the person in the picture and describe his or her life and important deeds.

Please be sure to follow these instructions when answering the questions:
- Please reply with zh-TW.
- If you don't know, just answer "I don't know." Please don't make inferences or answers at will.
- Solve and analyze problems, summarize the solution process and final answer, and explain the process.
- Please clearly mark the correct answer position (red) first, and please answer the explanation concisely without redundancy.
'''

if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if gemini_key is None:
    print('Specify GEMINI_API_KEY as environment variable.')
    sys.exit(1)

# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Initialize the Gemini Pro API
genai.configure(api_key=gemini_key)

#writing the api for flask api /api/image/search post request 
@app.post("/api/image/search")
async def search_image(request: Request):
    body = await request.body()
    print(body)
    img = PIL.Image.open(BytesIO(body))
    print(img)
    return google_image_search_content(body)

@app.post("/")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if (event.message.type == "text"):
            # Provide a default value for reply_msg
            msg = event.message.text
            ret = generate_gemini_text_complete(f'{msg}, reply in zh-TW:')
            reply_msg = TextSendMessage(text=ret.text)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
                
        elif (event.message.type == "image"):
            message_content = await line_bot_api.get_message_content(
                event.message.id)
            image_content = b''
            async for s in message_content.iter_content():
                image_content += s
            img = PIL.Image.open(BytesIO(image_content))
            # google_image_search_content(img)
            result = generate_result_from_image(img, images_prompt)
            reply_msg = TextSendMessage(text=result.text)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
            return 'OK'
        else:
            continue

    return 'OK'


def generate_gemini_text_complete(prompt):
    """
    Generate a text completion using the generative model.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response


def generate_result_from_image(img, prompt):
    """
    Generate a image vision result using the generative model.
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    return response

def google_image_search_content(img):
    text = image_to_text(img)
    print(f"圖片中的文字: {text}")
    results = google_search(text)
    print(results)
    parse_results(results['items'])  
    return results

def image_to_text(image):
    """將圖片轉換成文字"""
    """將 binary 圖片數據轉換為文字"""
    # 將 binary 數據轉換為 PIL Image 對象
    
    img = PIL.Image.open(BytesIO(image))
    # 使用 Tesseract 進行 OCR，並指定語言
    img = img.convert('L')
    text = pytesseract.image_to_string(img)  # 可以根據您的語言需求調整
    return text

def google_search(query):
    """使用 Google Custom Search API 搜尋"""
    # 替換成你的 Google Custom Search API 鍵
    api_key = userdata.get('SEARCH_API_KEY')
    cse_id = userdata.get('CSE_ID')
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}"
    response = requests.get(url)
    return response.json()

def parse_results(results):
    """解析搜尋結果"""
    for item in results:  # Iterate over the list directly
        print(f"標題: {item['title']}")
        print(f"連結: {item['link']}")
        print(f"描述: {item['snippet']}")
        print("-"*20)

