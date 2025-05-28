import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from browser_use import Agent

import asyncio
import os, io, requests, json
import base64
from PIL import Image
import numpy as np
import argparse

from datetime import datetime
from dateutil.parser import parse

from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
#from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config

from dotenv import load_dotenv
load_dotenv()

anime_styles= {"Jp 90s": "日本の90年代アニメ風。セル画のような色使いと質感で、太い輪郭線、大きな瞳、光沢のある髪のアニメスタイル",
        "Ghible": "ジブリ風",
        "Dragonquest": "鳥山明のドラゴンクエストスタイル"}
canvas_sizes= {"1024 × 1536": "1024 × 1536 (portrait orientation)",
        "4コマ": "4コマ",
        "ポスター": "ポスター"}

llms = {"chatgpt": "gpt-4o",
        "gemini": "gemini-2.0-flash-exp",
        "claude": "claude-3-5-sonnet-latest"}
options = {"""
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
"""}
default_model= "chatgpt"
default_steps= 10
default_name = "AniMaker"
default_cat  = "Book"
default_key  = "" #"AI_Key..."

shots = {"Wide": "wide shot",
        "Long": "long shot",
        "Medium": "medium shot",
        "Close": "close shot"
        }

from browser_use.browser.browser import Browser, BrowserConfig
browser = Browser(
	config=BrowserConfig(
		# headless=True,
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
	)
)

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--book', default="本か映画の名前を入れて!")
#args = parser.parse_args()
#book_name = args.book

search_type = ["New","Used","Rental", "Kindle","Audible"]
categories = ["Book","Movie","Music","Other"]
cat_options = {
    "Book": {"Amazon" : "https://amazon.co.jp",
        "Bookoff": "https://shopping.bookoff.co.jp/search",
        "Marcari": "https://jp.mercari.com/", 
        "Library": "https://setagayaliv.or.jp"},
        # "Rakuten": "https://rakuten.co.jp/"},
    "Movie": {"AmazonPrime": "https://www.amazon.co.jp/gp/video/storefront/ref=atv_hm_hom_legacy_redirect?contentId=IncludedwithPrime&contentType=merch&merchId=IncludedwithPrime",
        "Unext": "https://video.unext.jp/",
        "Netflix": "https://netflix.com"},
    "Other": {"Spotify": "https://spotify.com",
        "AmazonMusic": "https://music.amazon.co.jp"},
}

cats= []
def get_cat(category):
    cat_ops = []
    cat_urls= []
    for k1, v1 in cat_options.items():
        cats.append(k1)
        for k2, v2 in v1.items():
            if k1 == category:
                cat_ops.append(k2)
                cat_urls.append(v2)
    # print(cats, cat_ops, cat_urls)
    return cats, cat_ops, cat_urls

json_path= "./results/"
file_path= "./source/"

def ret_data(dates=5):
    hist_data = ""
    if os.path.isdir(json_path):
        json_files = os.listdir(json_path)
        for json_file in json_files:
            json_open = open(json_path+json_file, 'r')
            json_load = json.load(json_open)
            hist_date = parse(json_load['timestamp']).strftime('%Y/%m/%d %H:%M:%S')
            hist_data += hist_date + f" <a href='{json_path}{json_file}'>" + json_load['book'] + "</a><br>" #+json_load['data'] + "<br>"
    else:
        hist_data = "No history files"
    return hist_data

from openai import OpenAI
import google.generativeai as gemini
#from google.cloud import vision
#from google.oauth2 import service_account

async def save_data(book_data: str, book: str):
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "book": book,
        "data": book_data
    }
    filename = f'book_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open("./results/"+filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"\nデータを {filename} に保存しました")

async def main(chara_name, chara_out, image, general_style, llm_model, llm_key):
#def camera_detect(image,llm_model):

    llm_model=default_model
    apikey = os.getenv(llm_model+"_key")
    
    anime_prompt = f"""
# Prerequisites:
Title: {title}
Artist Requirements: {general_style}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose & Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] and reflecting the world of {general_style}
Please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
  - 主人公:
    - 名前：{chara_name}
    - 年齢、性別、髪型、表情、服装：{chara_out}
    - Please draw the protagonist with reference to the attached file `<file:shinji.png>`

- Overall setting:
    - Canvas size: {general_size}
    - Art style: {general_style} (used consistently in every panel)
    - Image quality: crisp and clear (used consistently in every panel)
    - Font: Noto Sans JP (used consistently in every panel)
    - Panel Margins: Each panel should have a uniform margin of 10px on all four sides internally (between the artwork and the panel border).
    - Panel Border: All panels should have a consistent border, for example, a 2px solid black line.
    - Gutter Width: The space (gutter) between panels should be uniform, for example, 20px horizontally and vertically.
    - Page Margins: The entire page (canvas) should have a uniform margin of 30px around the area where panels are laid out.

- panel layout
 - panel 1
  - 画面構図: ワイドショット。シンジは、足に血を流しながら、道路に体を横たえ、白いトライアスロン・バイクの下敷きになったまま、動けなかった。
  - 演出指示: オーストラリアの海沿いの道路。光は降り注ぎ明るいが、自転車と共に倒れるシンジは暗いイメージ。
  - テキスト要素:
    - ナレーション: シンジは倒れていた。
    - 効果音・書き文字: バーン（道路に倒れているシンジと自転車）
  - キャラクター表情:
    - 主人公: 道路に体を横たえ、アスファルトの堅いゴツゴツした感触に、妙に現実感がなく、呆然と倒れている。

 - panel 2
  - 画面構図: ロングショット。オーストラリア・ケアンズで行われているアイアンマン・レースの中ほど、バイクパートの90km付近を白いトライアスロン・バイクで走っているシンジ
  - 演出指示: 主人公には明るいスポットライトが当たっているような印象。背景にはケアンズの海とそこを真っ直ぐに走る直線的な道路。
  - テキスト要素:
    - ナレーション: アイアンマン・レースとは、スイム3.8km、バイク180km、ラン42.2kmの最長距離トライアスロン大会
  - キャラクター表情:
    - 主人公: 一生懸命、自転車を漕いでいる。

 - panel 3
  - 画面構図: ミディアムショット。海沿いのアップダウンが続く道の途中の登り。白いトライアスロン・バイクに乗ったシンジの足が攣ってしまう。
  - 演出指示: 立ちゴケしてショックを受けている
  - テキスト要素:
    - ナレーション: 突然両太モモが攣ってしまったのだ！
    - 効果音・書き文字: (足がつって衝撃的なイメージ) ビシー
  - キャラクター表情:
    - 主人公: 自転車に乗っている時に足が攣って、痛そうな顔

# supplement:
- Reconfirmation of instructions is not required.
- Self-evaluation is not required.
- Please output images only.

"""

    image_base64 = encode_image(image)
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "gemini":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, anime_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "chatgpt":
        openai_client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = openai_client.chat.completions.create(
            model=llms[llm_model],
            messages = [
                {'role': 'system',
                    'content': "このシステムは、画像が提供された時に、その画像の中に何が写っているかを判別します。"},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                            'text': anime_prompt},
                        {'type': 'image_url',
                            'image_url': {'url': image_base64}},
                    ]
                },
            ],
            temperature = 0.9,
            max_tokens = 256,
        )
        result = response.choices[0].message.content
        print(result)
        return result
    

#from google.cloud import vision
#from google.oauth2 import service_account
#import base64

#async def main(booka_out,category, llm_model,num_steps,llm_key):
async def main2(chara_name, chara_out, img_up, general_style, llm_model, llm_key):

    cat_ret = get_cat(category)
    if llm_key:
        apikey = llm_key
    else:
        apikey = os.getenv(llm_model+"_key")

    if llm_model == "chatgpt":
        llm_api = ChatOpenAI(model=llms[llm_model], api_key=apikey)
    elif llm_model == "gemini":
        llm_api = ChatGoogleGenerativeAI(model=llms[llm_model], api_key=apikey)
    elif llm_model == "claude":
        llm_api = ChatAnthropic(model_name=llms[llm_model], api_key=apikey)
    LLM_MODEL = llm_model.upper()

    agent = Agent(
        task=f""" ABC """,
        llm=llm_api,
		#controller=controller,
		browser=browser,
    )
    #info   = await book_info(book)
    history= await agent.run(max_steps=num_steps)
    result = history.final_result()
    print(result)
    await save_data(result, booka_out)
    return result


def encode_image(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    base64_image = f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    return base64_image

def camera_detect(image,llm_model):
    llm_model=default_model
    """if llm_key:
        apikey = llm_key
    else:"""
    apikey = os.getenv(llm_model+"_key")
    camera_prompt = "提供された画像の中に写っている人物の、おおよその年齢、性別を類推して下さい。髪型、表情、服装を詳細に簡潔に説明して下さい。"

    image_base64 = encode_image(image)
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "gemini":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, camera_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "chatgpt":
        openai_client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = openai_client.chat.completions.create(
            model=llms[llm_model],
            messages = [
                {'role': 'system',
                    'content': "このシステムは、画像が提供された時に、その画像の中に何が写っているかを判別します。"},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                            'text': camera_prompt},
                        {'type': 'image_url',
                            'image_url': {'url': image_base64}},
                    ]
                },
            ],
            temperature = 0.9,
            max_tokens = 256,
        )
        result = response.choices[0].message.content
        print(result)
        return result
    
    """
    print(result)
    nTitle = re.search(r'題名|タイトル|書籍名|Title', result)
    if nTitle:
        result = result[nTitle.end():]
    #chat_msg = "BOOKA Name: " + result
    """
    #return result

#img_up = gr.Image(label="Book Photo", sources="webcam",webcam_constraints=["rear"], 
#                  type="pil", mirror_webcam=False, width=350,height=350)
img_up = gr.Image(label="Chara Photo", sources="upload",
                  type="pil", mirror_webcam=False, width=300,height=300)

#img_out= gr.Textbox(label="Book Name")

def flip(im):
    return np.fliplr(im)

#cat_ret = get_cat("Book")
with gr.Blocks() as demo:
    #gr.Markdown("<h1>Booka - Price Search App</h1>")
    with gr.Row():
        with gr.Column():
            with gr.Tab(default_name+"!"):
                with gr.Row():
                    general_style= gr.Dropdown(choices=anime_styles,label="Anime Style", interactive=True)
                    general_size = gr.Dropdown(choices=canvas_sizes,label="Canvas size", interactive=True)
                    llm_model    = gr.Dropdown(choices=llms,label="LLM", interactive=True)
                    chara_name= gr.Textbox(label="Chara Name", interactive=True)
                    chara_out = gr.Textbox(label="Outfit",placeholder="Character's outfit",interactive=True)
                    camera    = gr.Interface(fn=camera_detect,
                        inputs=[img_up,llm_model], outputs=chara_out, live=True, 
                        flagging_mode="never", clear_btn=None)
                    
                output = gr.Markdown("Panel layout:")
                with gr.Row():
                    panel_shot= gr.Dropdown(choices=shots,label="Panel1 Shot", interactive=True)
                    panel_name = gr.Textbox(label="Name", interactive=True)
                    panel_naration= gr.Textbox(label="Naration", interactive=True)
                    panel_sound   = gr.Textbox(label="Sound", interactive=True)
                    panel_content = gr.Textbox(label="Content", interactive=True)
                
                with gr.Row():
                    panel_shot= gr.Dropdown(choices=shots,label="Panel2 Shot", interactive=True)
                    panel_name = gr.Textbox(label="Name", interactive=True)
                    panel_naration= gr.Textbox(label="Naration", interactive=True)
                    panel_sound   = gr.Textbox(label="Sound", interactive=True)
                    panel_content = gr.Textbox(label="Content", interactive=True)


            with gr.Tab("All Other Anime"):
                with gr.Row():
                    category = gr.Dropdown(choices=categories,label="Category",value=default_cat,interactive=True)
                    booka_out= gr.Textbox(label="Name", placeholder="Put Book name, Movie, or any goods here")
                    # auther = gr.Textbox(label="Auther",placeholder="荒木飛呂彦")
                    # isbn   = gr.Textbox(label="ISBN/ASIN",placeholder="")
                    # bookpic= gr.Button("Book Photo (Launch camera)")
        with gr.Column():
            with gr.Accordion(label="Search options:", open=False):
                with gr.Column():
                    with gr.Tab("Book"):
                          gr.CheckboxGroup(get_cat("Book")[1], label="Book search for:", interactive=True,
                            value=get_cat("Book")[1])
                          gr.Textbox(label="Additional search site:", placeholder="https://...",interactive=True,)
                          gr.CheckboxGroup(search_type,
                            label="Including:", interactive=True,
                            value=search_type)
                    with gr.Tab("Others"):
                          gr.CheckboxGroup(get_cat("Movie")[1],
                            label="Movie search for:", interactive=True,
                            value=[])
                          gr.CheckboxGroup(get_cat("Other")[1],
                            label="Other search for:", interactive=True,
                            value=[])
                          gr.Textbox(label="Additional search site:", placeholder="https://...",interactive=True,)
                #llm_model= gr.Dropdown(choices=llms,label="LLM",value=default_model, interactive=True)
                llm_key  = gr.Textbox(label="LLM API Key",value=default_key,placeholder="Paste your LLM API key here", interactive=True,)
                num_steps= gr.Slider(minimum=1,maximum=20,value=default_steps,step=1, label="Steps",interactive=True)

    output = gr.Markdown("ANIME Reuslt:")
    search_btn = gr.Button("AniMake!")
    search_btn.click(fn=main, inputs=[chara_name, chara_out, img_up, general_style, llm_model, llm_key], outputs=output, api_name="animaker")

    with gr.Accordion(label="Anime history:", open=False):
        hist_data = ret_data(5)
        gr.Markdown(hist_data)

parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
args = parser.parse_args()

demo.launch(server_name=args.ip,server_port=args.port) #, auth=("usr","pswd"))
