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

page_styles= {"Jp 90s": "日本の90年代アニメ風。セル画のような色使いと質感で、太い輪郭線、大きな瞳、光沢のある髪のアニメスタイル",
        "Ghible": "ジブリ風",
        "Dragonquest": "鳥山明のドラゴンクエストスタイル"}
page_sizes= {"1024 × 1536": "1024 × 1536 (portrait orientation)",
        "4コマ": "4コマ",
        "ポスター": "ポスター"}
page_storys= {"Original": "主人公を中心として、入力されたページ構成を変更せず、忠実に従って下さい。",
        "Generate": "主人公を中心としたストーリーを、ページ構成に従って、できる限り詳細に生成して下さい。",
        "Hybrid": "主人公を中心としたストーリーを、入力された情報に付加して、作り上げて下さい。"}

llms = {"chatgpt": "gpt-image-1", #"gpt-4o",
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

panel_sizes = {"Small": "Small size",
        "Medium": "Medium size",
        "Big": "Big size",
        "Very big": "Very big size"
        }
panel_shots = {"Full": "Full shot",
        "Wide": "Wide shot",
        "Knee": "Knee shot",
        "Waist": "WWaist shot",
        "Up": "Up shot",
        "Close-up": "Close-up shot"
        }

results_path = './results/'
imgup_path= "./img_ups/"

#from browser_use.browser.browser import Browser, BrowserConfig
#browser = Browser(
#	config=BrowserConfig(
#		# headless=True,
#        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
#	)
#)

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
    }
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

def ret_data(dates=5):
    hist_data = ""
    if os.path.isdir(results_path):
        files = os.listdir(results_path)
        for file in files:
            json_open = open(results_path+file, 'r')
            json_load = json.load(json_open)
            #hist_date = parse(json_load['timestamp']).strftime('%Y/%m/%d %H:%M:%S')
            hist_data += f" <a href='{results_path}{file}'>" + json_load + "</a><br>"
    else:
        hist_data = "No history files"
    return hist_data

from openai import OpenAI
import google.generativeai as gemini
#from google.cloud import vision
#from google.oauth2 import service_account

async def save_data(chara_name: str, image_data: str):
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "chara": chara_name,
        "data": image_data
    }
    filename = f'{chara_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open("./results/"+filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"\nデータを {filename} に保存しました")

async def main(page_title, page_style, page_size, chara_name, chara_out, img_up, llm_model,
            panel1_shot,panel1_comp,panel1_naration,panel1_others, 
            panel2_shot,panel2_comp,panel2_naration,panel2_others,
            panel3_shot,panel3_comp,panel3_naration,panel3_others, 
            panel4_shot,panel4_comp,panel4_naration,panel4_others):
    llm_model=default_model
    apikey = os.getenv(llm_model+"_key")

    anime_prompt = f"""
# Prerequisites:
Title: {page_title}
Artist Requirements: {page_styles[page_style]}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose & Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] and reflecting the world of {page_styles[page_style]}
Please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
  - 主人公:
    - 名前：{chara_name}
    - 年齢、性別、髪型、表情、服装：{chara_out}
    - Please draw the protagonist with reference to the attached file `<file:img_ups/image.jpg>`

- Overall setting:
    - Canvas size: {page_sizes[page_size]}
    - Art style: {page_styles[page_style]} (used consistently in every panel)
    - Image quality: crisp and clear (used consistently in every panel)
    - Font: Noto Sans JP (used consistently in every panel)
    - Panel Margins: Each panel should have a uniform margin of 10px on all four sides internally (between the artwork and the panel border).
    - Panel Border: All panels should have a consistent border, for example, a 2px solid black line.
    - Gutter Width: The space (gutter) between panels should be uniform, for example, 20px horizontally and vertically.
    - Page Margins: The entire page (canvas) should have a uniform margin of 30px around the area where panels are laid out.

- Page Title: {page_title}

# supplement:
- Reconfirmation of instructions is not required.
- Self-evaluation is not required.
- Please output images only.
"""
    
    original_prompt = f"""
- Panel layout:
 - Panel 1 
  - Panel composition: {panel1_shot}
  - 演出指示:. {panel1_comp}
  - テキスト要素:
    - ナレーション: {panel1_naration}
    - 効果音・書き文字: このPanel1のシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanel1のシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: {panel1_others}
    
 - Panel 2
  - Panel composition: {panel2_shot}
  - 演出指示:. {panel2_comp}
  - テキスト要素:
    - ナレーション: {panel2_naration}
    - 効果音・書き文字: このPanel2のシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanel2のシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: {panel2_others}

 - Panel 3
  - Panel composition: {panel3_shot}
  - 演出指示:. {panel3_comp}
  - テキスト要素:
    - ナレーション: {panel3_naration}
    - 効果音・書き文字: このPanel3のシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanel3のシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: {panel3_others}

 - Panel 4
  - Panel composition: {panel4_shot}
  - 演出指示:. {panel4_comp}
  - テキスト要素:
    - ナレーション: {panel4_naration}
    - 効果音・書き文字: このPanel3のシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanel3のシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: {panel4_others}

"""
    
    if page_story == "generate":
        anime_prompt = anime_prompt + generate_prompt
    else:
        anime_prompt = anime_prompt + original_prompt

    image_base64 = encode_image(img_up) # open(img_up, "rb")
    #filename = f'{chara_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    #imagefile = './img_ups/'+filename+'.jpg'
    #imagefile = img_up.filename
    #print(imagefile)
    #with open(imagefile, "wb") as f:
    #    f.write(base64.b64decode(response.data[0].b64_json))
    #with open(image_path, "rb") as image_file:
    #    return base64.b64encode(image_file.read()).decode('utf-8')
    source_image = open('./img_ups/image.jpg', "rb")

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
        client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = client.images.edit(
            model = "gpt-image-1",
            image = source_image,
            prompt= anime_prompt
        )
        image_response = response.data[0].b64_json

        #output_data = {
            #"timestamp": datetime.now().isoformat(),
            #"chara": chara_name,
            #"data": image_data
        #}
        
        filename = f'{chara_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        imagefile = results_path+'image_'+filename+'.jpg'
        promptfile= results_path+'prompt_'+filename+'.txt'

        with open(promptfile, 'a', encoding='utf-8') as f:
            f.write(anime_prompt)
            #json.dump(anime_prompt, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',',': '))

        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
            #image = Image.open(BytesIO(part.inline_data.data))
            #image_path = f"results/edit_{image_count}.jpg"
            #image.save(image_path)
        
        print(imagefile+" created!")
    
        return imagefile

#from google.cloud import vision
#from google.oauth2 import service_account
#import base64

#async def main(booka_out,category, llm_model,num_steps,llm_key):
async def main2(chara_name, chara_out, img_up, page_style, llm_model, llm_key):

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

def camera_nodetect(image):
    image.save("./img_ups/image.jpg")
    return "please put image chara info: "

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

    image.save("./img_ups/image.jpg")

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
            model="gpt-4o-mini", #llms[llm_model],
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

#img_up = gr.Image(label="Book Photo", sources="webcam",webcam_constraints=["rear"], 
#                  type="pil", mirror_webcam=False, width=350,height=350)
img_up = gr.Image(label="Chara Photo", sources="upload",
                  type="pil", mirror_webcam=False, width=300,height=300)

def flip(im):
    return np.fliplr(im)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tab(default_name+"!"):
                with gr.Row():
                    page_style= gr.Dropdown(choices=page_styles,label="Anime Style", interactive=True)
                    page_size = gr.Dropdown(choices=page_sizes,label="Canvas size", interactive=True)
                    page_story= gr.Dropdown(choices=page_storys,label="Original/Generate", interactive=True)
                    chara_name= gr.Textbox(label="Chara Name", interactive=True)
                    page_title= gr.Textbox(label="Title", interactive=True)
                    page_plot = gr.Textbox(label="Plot", interactive=True)
                    chara_out = gr.Textbox(label="Charactor",placeholder="Upload photo and edit results",interactive=True)
                    camera    = gr.Interface(fn=camera_nodetect, #camera_detect,
                        inputs=[img_up], outputs=chara_out, live=True, 
                        flagging_mode="never", clear_btn=None)
                    
                output = gr.Markdown("Panel layout:")
                with gr.Row():
                    #panel1_size= gr.Dropdown(choices=panel_sizes,label="Panel1 Size", interactive=True)
                    panel1_shot= gr.Dropdown(choices=panel_shots,label="Panel 1 Shot", interactive=True)
                    panel1_comp= gr.Textbox(label="Composition", interactive=True)
                    panel1_naration = gr.Textbox(label="Naration", interactive=True)
                    #panel1_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                    #panel1_face  = gr.Textbox(label="Face", interactive=True)
                    panel1_others= gr.Textbox(label="Other charactors", interactive=True)                

                with gr.Row():
                    #panel2_size= gr.Dropdown(choices=panel_sizes,label="Panel2 Size", interactive=True)
                    panel2_shot= gr.Dropdown(choices=panel_shots,label="Panel 2 Shot", interactive=True)
                    panel2_comp= gr.Textbox(label="Composition", interactive=True)
                    panel2_naration = gr.Textbox(label="Naration", interactive=True)
                    #panel2_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                    #panel2_face = gr.Textbox(label="Face", interactive=True)
                    panel2_others= gr.Textbox(label="Others charactors", interactive=True)

                with gr.Row():
                    #panel3_size= gr.Dropdown(choices=panel_sizes,label="Panel3 Size", interactive=True)
                    panel3_shot= gr.Dropdown(choices=panel_shots,label="Panel 3 Shot", interactive=True)
                    panel3_comp= gr.Textbox(label="Composition", interactive=True)
                    panel3_naration = gr.Textbox(label="Naration", interactive=True)
                    #panel3_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                    #panel3_face = gr.Textbox(label="Face", interactive=True)
                    panel3_others= gr.Textbox(label="Others charactors", interactive=True) 

                with gr.Row():
                    #panel4_size= gr.Dropdown(choices=panel_sizes,label="Panel4 Size", interactive=True)
                    panel4_shot= gr.Dropdown(choices=panel_shots,label="Panel 4 Shot", interactive=True)
                    panel4_comp= gr.Textbox(label="Composition", interactive=True)
                    panel4_naration = gr.Textbox(label="Naration", interactive=True)
                    #panel4_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                    #panel4_face = gr.Textbox(label="Face", interactive=True)
                    panel4_others= gr.Textbox(label="Other charactors", interactive=True)

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
                llm_model    = gr.Dropdown(choices=llms,label="LLM", interactive=True)
                llm_key  = gr.Textbox(label="LLM API Key",value=default_key,placeholder="Paste your LLM API key here", interactive=True,)
                num_steps= gr.Slider(minimum=1,maximum=20,value=default_steps,step=1, label="Steps",interactive=True)

    #output = gr.Markdown("ANIME Reuslt:")
    search_btn = gr.Button("AniMake!")
    search_btn.click(fn=main, inputs=[page_title, page_style, page_size, chara_name, chara_out, img_up, llm_model, 
                panel1_shot,panel1_comp,panel1_naration,panel1_others, 
                panel2_shot,panel2_comp,panel2_naration,panel2_others,
                panel3_shot,panel3_comp,panel3_naration,panel3_others, 
                panel4_shot,panel4_comp,panel4_naration,panel4_others],
                outputs=[
                    gr.Image(
                        label="ANIME Result:",
                    #gr.Gallery(
                        #show_label=True,
                        #elem_id="Gallery",
                        #columns=[2],
                        #object_fit="contain",
                        #height="auto",
                    ),
                    #gr.Textbox(label="Error Messages"),
                ], api_name="animaker")
    
    #with gr.Accordion(label="Anime history:", open=False):
        #hist_data = ret_data(5)
        #gr.Markdown(hist_data)

parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
args = parser.parse_args()

demo.launch(server_name=args.ip,server_port=args.port) #, auth=("usr","pswd"))
