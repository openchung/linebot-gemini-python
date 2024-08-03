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
image_prompt = '''
您是圖片分析專家。請用科學的細節描述這張圖片並分析圖片的類型。
影像類型分析處理方法如下：
1. 人物資訊
翻譯成中文並以科學細節解釋
2. 資訊傳達類型
翻譯成中文並以科學細節解釋
2. 問題類型
解決和分析問題，總結解決過程和最終答案，並解釋過程
最後請用zh-TW產生問題內容描述

將上述得出的問題內容描述，當作問題，並經過下面的循環，直到得出最終答案：

你在 Thought、Action、PAUSE、Observation 的循環中運行。
在循環結束時，你輸出 Answer。
使用 Thought 描述你對上述問題的想法。
使用 Action 執行你可以使用的行動之一，然後返回 PAUSE。
Observation 將是執行這些行動的結果。

你可以使用的行動有：

google_image_search_content:


fetch_ticker:
找出一段文字中所描述的金融商品、標的或是整個市場
例如：fetch_ticker： 一段文字"今天 CPI 低於預期" 標的為"市場"
     fetch_ticker: 一段文字"台積電今天不太行" 標的為"台積電"

fetch_stock_data:
例如 fetch_stock_data: 台積電
台積電在yfinance的代號為 2330.tw
查詢近期股價變化

analyze_sentiment:
例如 analyze_sentiment: 台積電
以"正面"、"負面"、"中性"的三種結果分析一段關於金融市場的情緒
例如：analyze_sentiment: 一段文字"台積電今天不太行" 是"負面"的
Runs a analyze_sentiment and returns results

範例對話：

Question: 台積電將調高資本資出
Thought: 這句話的金融標的為何
Action: 分析標的: 台積電將調高資本資出
PAUSE

這時會返回：

Observation: 這句話的標的為"台積電"

接下來你會執行：

Action: fetch_stock_data: 台積電
台積電在 yfinance 的代號為 2330.tw
PAUSE

Observation: 台積電最近五天股價變化（例如：-20, -10, 0, 20）

接下來你會執行：

Action: analyze_sentiment: 最近五天股價變化為（例如：-20, -10, 0, 20），"台積電將調高資本資出"的情緒為?
PAUSE

最後你輸出：

Answer: 標的：台積電，情緒：正面，股價變化：例如：-20, -10, 0, 20）
'''

images_prompt = '''
You are an expert in picture analysis. Please describe this picture in scientific details and analyze the type of picture.
The image type analysis and processing methods are as follows:
1. Information type
Translated into Chinese and explained in scientific detail
2. Question type
Solve and analyze problems, summarize the solution process and final answer, and explain the process
Finally, please respond in zh-TW:
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

