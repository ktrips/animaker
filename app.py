import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_anthropic import ChatAnthropic
#from browser_use import Agent
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
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

page_styles= {"Jp 90s": "日本の90年代アニメ風。セル画のような色使いと質感で、太い輪郭線、大きな瞳、光沢のある髪のアニメスタイル",
        "Ghibli": "スタジオジブリ風のアニメイラストに変換。自然豊かな背景、柔らかい線画、温かみのある色合いを表現。",
        "Pixar": "ディズニー・ピクサー風の3Dアニメスタイルに変換。大きな瞳、立体的な質感、明るい色合いで描画。",
        "Jojo": "アニメのジョジョ風（ジョジョの奇妙な冒険）に変換。太い輪郭線、激しい陰影、ポーズを強調し、カラフルな背景でジョジョの奇妙な冒険風にする。",
        "Dragonquest": "ドラゴンクエスト風に変換。鳥山明風のイラストで、剣や魔法、明るいファンタジー世界の背景、ポップな表情で描いてください。",
        "Slumdank": "アニメのスラムダンク風に変換してください。ユニフォームを着て、汗や動きのあるポーズ、青春漫画風のリアルな線画で描いてください。",
        "Pokemon": "ポケモン風のキャラクターデザインに変換。アニメタッチでかわいらしく、明るい色合いで仕上げ。",
        "Eva": "エヴァンゲリオン風に変換。モダンなメカニカルデザインと、ダークでシリアスな色調を追加。",
        "Kimetsu": "鬼滅の刃風に変換。和装デザインと、呼吸技のエフェクトを加えたスタイリッシュなタッチにする。",
        "Akira": "アニメのAKIRA風に変換。ネオ東京の荒廃した都市を背景に、赤いバイクに乗ったキャラクターを描く。赤や黒を基調とした色使いで、ネオンライトや未来的な都市の雰囲気を追加。",
        "Ukiyoe": "浮世絵の葛飾北斎風に変換。太い輪郭線と落ち着いた和風色彩で描き、背景には浮世絵スタイルの山や波を加える。",
        "Retro": "レトロゲーム風に変換。8bitドット絵スタイルで、カラフルかつレトロな雰囲気にする。",
        "Chikawa": "ちいかわ風に変換してください。丸くて小さな体型、シンプルな線画、ほのぼのした背景で描いてください。"}

page_sizes= {"512×768": "512 × 768 (portrait orientation)",
            "1024x1536": "1024 × 1536 (portrait orientation)",
            "4コマ": "4コマ",
            "ポスター": "ポスター"}
page_storys= {"Manual": "主人公を中心として、入力されたページ構成を変更せず、忠実に従って下さい。",
        "Generate": "主人公を中心としたストーリーを、ページ構成に従って、できる限り詳細に生成して下さい。",
        "Hybrid": "主人公を中心としたストーリーを、入力された情報に付加して、作り上げて下さい。"}
image_qualities= ["low","medium","high"]
generate_pages = [1,2,3,4,5]
page_panels    = [1,2,3,4,5]

colors = ["指定なし","黒","茶","赤","青","黄","緑","紫","ピンク","オレンジ","白"]

llms = {"OPENAI_API": "gpt-4o-mini", #"gpt-image-1", #,
        "GOOGLE_API": "gemini-2.0-flash-exp",
        "ANTHOLOPIC": "claude-3-5-sonnet-latest"}
options = {"""
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
"""}

default_model= "OPENAI_API"
#default_key  = "AI_Key..."
default_style= "Jp 90s"
default_story= "Generate"
default_size   = "1024x1536" #"1536x1024" #"1024x1024"
default_quality= "high"
default_page   = 1
default_panel  = 5

default_color = "指定なし"

default_steps= 10
default_cat  = "Book"

#result_folder= "./results"
#up_folder    = "./up"
results_path = './results/'
image_path = "./image/"
img_up_path= image_path+'img_up.jpg'
chara_path = "./chara/"

default_chara= "シンジ"
charas = {
    "シンジ": ["Shinji", chara_path+"shinji_anime.jpg","赤いトライアスロンウェアに身を包んだ、30代中肉中背の男性。髪は短く、黒色の髪。ニヤリと笑っている","赤"],
    "ノリコ": ["Noriko", chara_path+"noriko_anime.jpg","ブルーのランニングウェアに身を包んだ、ギャルっぽい女の子。髪はショートで青色の髪。スポーティーで笑顔が可愛い","青"],
    "ゴトー": ["Goto",   chara_path+"goto_anime.jpg","黒いウェアにサングラスをかけている。40代の男性で鬼軍曹のような厳しい雰囲気。髪はベリーショートで、真っ黒","黒"],
    "ハラダ": ["Harada", chara_path+"harada_anime.jpg","オレンジのトライアスロンウェアを着た、40代の若手経営者。髪はお洒落なパーマで金髪。キラキラして自信ありげな笑みを浮かべる","金"],
    "マツイ": ["Matsui", chara_path+"matsui_anime.jpg","白いランニングウェアを着たマッチョな30代男性。寡黙だが子煩悩なパパでもある。髪の毛はミディアムで緑色","紫"],
    "ケン":   ["Ken",    chara_path+"ken_anime.jpg","緑のトライアスロンウェアを着た痩せぎすな30代男性。自信なさげだが内なる闘志を秘めている。髪はサラサラで赤色","緑"],
    "ナオト": ["Naoto",  chara_path+"naoto_anime.jpg","ナオトは世界中を旅している20代の大学生。背は低いが、足が速く、引き締まった体をしている","青"],
    "ケニー":  ["Kenny", chara_path+"kenny_anime.jpg","ケニーは世界中を旅している20代の大学生。引き締まった体をしている","赤"],
    "New":   ["new",  chara_path+"others_anime.jpg","それぞれメンバーの仲間たち","黒"],
}

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

def ret_data(nums):
    image_list = []
    if os.path.isdir(results_path):
        files= os.listdir(results_path)
        #sorted_file_names = sorted(file_names)
        for file in sorted(files):
            #file_open = open(results_path+file, 'r')
            #file_load = json.load(file_open)
            #hist_date = parse(json_load['timestamp']).strftime('%Y/%m/%d %H:%M:%S')
            if file[-4:] == ".jpg":
                image_list.append(results_path+file)
                #f"<a href='{results_path}{file}'><img href='{results_path}{file}' width=100></a> "
    else:
        image_list = "No image files"
    print(image_list)
    return image_list

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

async def add_chara(chara_up, chara_name, chara_desc, chara_color): #, prompt_out, #img_up, llm_model, prompt_out):
    print("== Chara Generation ==\n Starting Anime image generation!\n")
    llm_model=default_model
    #print(llm_model)
    #print(prompt_out)
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(llm_model+"_KEY")

    chara_img = chara_path+chara_name+".jpg"
    chara_up.save(chara_path)
    source_image = open(chara_img, "rb")
    image_base64 = encode_image(chara_up) # open(img_up, "rb")

    anime_prompt= f"""ここに写っている人物を、日本の90年代アニメ風の画像に変換して下さい。
        セル画のような色使いと質感で、太い輪郭線、光沢のある髪のアニメスタイルです。画像のみを出力して下さい。
        顔や表情は写真に正確に忠実に表現して下さい。
        この人物のイメージは、{chara_desc}で、髪の毛は{chara_color}色で印象的な髪型に変更して下さい。服装も全体的に{chara_color}色風にして下さい。
        ロゴや文字などは架空のものに変更して下さい。
        背景は海沿いの青空にして下さい。"""

    if llm_model == "GOOGLE_API":
        generate_model = llms[llm_model]
        gemini.configure(api_key=apikey)
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048}
        gemini_client = gemini.GenerativeModel(generate_model, generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, anime_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
        generate_model = "gpt-image-1"
        client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = client.images.edit(
            model  = generate_model, #llms[llm_model], #"gpt-image-1",
            image  = source_image,
            prompt = anime_prompt,
            quality= default_quality,
            size   = default_size,
            #n = generate_pages
        )
        image_response = response.data[0].b64_json
        #filename  = "chara_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        imagefile = chara_path+chara_name+'_anime.jpg'
        #with open(results_path+filename+'_prompt.txt', 'a', encoding='utf-8') as f:
                #f.write(prompt_out)
                #json.dump(anime_prompt, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',',': '))
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))

        new_chara_set = {chara_name: [chara_name, imagefile, chara_desc, chara_color]}
        print(new_chara_set)
        #charas.append(new_chara_set)
    
        return imagefile

async def main(prompt_out): #img_up, llm_model, prompt_out):
    print("== Main Generation ==\n Starting Anime image generation!\n")
    llm_model=default_model
    #print(llm_model)
    #print(prompt_out)
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(llm_model+"_KEY")
    source_image = open(img_up_path, "rb")
    #image_base64 = encode_image(source_image) # open(img_up, "rb")

    if llm_model == "GOOGLE_API":
        generate_model = llms[llm_model]
        gemini.configure(api_key=apikey)
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048}
        gemini_client = gemini.GenerativeModel(generate_model, generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, prompt_out])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
        generate_model = "gpt-image-1"
        client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = client.images.edit(
            model  = generate_model, #llms[llm_model], #"gpt-image-1",
            image  = source_image,
            prompt = prompt_out,
            quality= default_quality,
            size   = default_size,
            #n = generate_pages
        )

        image_response = response.data[0].b64_json
        filename  = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        imagefile = results_path+filename+'_image.jpg'
        with open(results_path+filename+'_prompt.txt', 'a', encoding='utf-8') as f:
                f.write(prompt_out)
                #json.dump(anime_prompt, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',',': '))
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
            #image = Image.open(BytesIO(part.inline_data.data))
            #image_path = f"results/edit_{image_count}.jpg"
            #image.save(image_path)
        print(imagefile+" created!")
    
        return imagefile
    

async def main3(page_title, page_style, chara_name, chara_out, img_up, llm_model,
            panel1_shot,panel1_comp,panel1_naration,panel1_others, 
            panel2_shot,panel2_comp,panel2_naration,panel2_others,
            panel3_shot,panel3_comp,panel3_naration,panel3_others, 
            panel4_shot,panel4_comp,panel4_naration,panel4_others):
    #llm_model=default_model
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
    - Please draw the protagonist with reference to the attached file `<file:{img_up_path}>`

- Overall setting:
    - Canvas size: {default_size}
    - Art style: {page_styles[page_style]} (used consistently in every panel)
    - quality: {default_quality}
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
    source_image = open(img_up, "rb")

    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "GOOGLE_API":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, anime_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
        client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = client.images.edit(
            model  = llms[llm_model], #"gpt-image-1",
            image  = source_image,
            prompt = anime_prompt,
            quality= default_quality,
            size   = default_size,
            #n = generate_pages
        )
        image_response = response.data[0].b64_json
        
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

def plot_generate(img_up, chara_name, page_plot, title_plot):
    print(f"== Prompt Generation ==\n Starting Prompt creation for {chara_name} from Plot!\n")
    llm_model=default_model
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(llm_model+"_KEY")
    openai_client = OpenAI(api_key=apikey)

    img_up.save(img_up_path)
    image_base64 = encode_image(img_up)

    if title_plot == "Title":
        response = openai_client.chat.completions.create(
            model=llms[llm_model],
            messages = [
                {'role': 'system',
                    'content': "このシステムは、テキストが提供された時に、それをタイトルとした物語のあらすじを3００字程度で、起承転結を付けて生成します。"},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                         'text': page_plot},
                        {'type': 'image_url',
                         'image_url': {'url': image_base64}},
                    ]
                },
            ],
            temperature = 0.9,
            max_tokens = 2048, #256,
        )
        plot_result = response.choices[0].message.content
        print("Plot from Title: "+plot_result)
        use_plot = plot_result
    else:
        use_plot = page_plot

    plot_prompt = f"""
「{chara_name}」はこの話の主人公です。この主人公は、{charas[chara_name][2]}のような風貌で、{charas[chara_name][3]}色の髪の毛、そのシーンに合った服装、表情をした人物です。
プロット「{use_plot}」から、5つのPanelを作り、各Panel毎に以下のフォーマットに従って、5つのpanelを記述したプロンプトを作成してください。
プロットに記述したランドマークや商品はなるべく正確に写実的に表現して絵柄に入れて下さい。
各Panelのフォーマット。このフォーマットを元に5つのPanelを作って下さい。
- Panel layout:
 - Panel
  - Panel composition: このPanelにピッタリなショットを、フルショット、ワイドショット、ミディアムショット、クローズアップショットから選んで下さい。
  - 演出指示:. このPanelにピッタリ合った演出表示を正確に記述して下さい。
  - テキスト要素:
    - ナレーション: このPanelにぴったり合うナレーションを生成して下さい。
    - 効果音・書き文字: このPanelのシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanelのシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: もし他の登場人物がいたら、表示して下さい。
    """

    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "GOOGLE_API":
        gemini.configure(api_key=apikey) #os.getenv("gemini_key"))
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, plot_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
        #openai_client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = openai_client.chat.completions.create(
            model=llms[llm_model],
            messages = [
                {'role': 'system',
                    'content': "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                            'text': plot_prompt},
                        {'type': 'image_url',
                            'image_url': {'url': image_base64}},
                    ]
                },
            ],
            temperature = 0.9,
            max_tokens = 2048, #256,
        )
        result = response.choices[0].message.content
        print("Generated Plot: "+result)

        anime_prompt = f"""
# Prerequisites:
Title: If the page plot 「{page_plot}」 includes "Title" then set it as the page title. If "Title" is not included do not set the title for this page.
Artist Requirements: {page_styles[default_style]}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose and Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] and reflecting the world of {page_styles[default_style]}
Please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
  - 主人公:
    - 名前: {chara_name}
    - Please draw the protagonist precisely with reference to the attached file `<file:{img_up_path}>`
    - 年齢、性別: Please estimate the protagonist's age and gender based on the attached file `<file:{img_up_path}>`
    - 髪型: {chara_name}は、{charas[chara_name][3]}色の髪の毛をしています。
    - 風貌: {charas[chara_name][2]}ような風貌です。
    - 服装: {chara_name}の服装は、そのシーンに合ったものです。
    - 表情: {chara_name}の表情は、そのシーンに合ったものです。

- Overall setting:
    - Canvas size: {default_size}
    - Art style: {page_styles[default_style]} (used consistently in every panel)
    - quality: {default_quality}
    - Image quality: crisp and clear (used consistently in every panel)
    - Font: Noto Sans JP (used consistently in every panel)
    - Panel Margins: Each panel should have a uniform margin of 10px on all four sides internally (between the artwork and the panel border).
    - Panel Border: All panels should have a consistent border, for example, a 2px solid black line.
    - Gutter Width: The space (gutter) between panels should be uniform, for example, 20px horizontally and vertically.
    - Page Margins: The entire page (canvas) should have a uniform margin of 30px around the area where panels are laid out.

# supplement:
- Reconfirmation of instructions is not required.
- Self-evaluation is not required.
- Please output images only.

- Page generation:
    - Page title: If the page plot includes "Title" then set it as the page title.
    - Please generate one page image exactly following the page and panel layouts based on [##Page layout] with your best effort.

## Page layout:

""" + result

        return anime_prompt


async def plot_image_generate(img_up,chara_name,page_plot,page_panel):
    print(f"== Prompt Image Generation ==\n From prompt directly generate Image for {chara_name}!\n")
    llm_model=default_model
    #print(llm_model)
    #print(prompt_out)
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(llm_model+"_KEY")

    if len(page_plot) < 40:
        page_plot = plot_generate(img_up,chara_name,page_plot,"Title")

    if chara_name == "New":
        img_up = add_chara(img_up, chara_name, page_plot, charas[chara_name][3])

    plot_prompt = f"""
「{chara_name}」はこの話の主人公です。この主人公は、{charas[chara_name][2]}のような風貌で、{charas[chara_name][3]}色の髪の毛、そのシーンに合った服装、表情をした人物です。
プロット「{page_plot}」から、{page_panel}つのPanelを作り、各Panel毎に以下のフォーマットに従って、{page_panel}つのpanelを記述したプロンプトを作成してください。
プロットに記述したランドマークや商品はなるべく正確に写実的に表現して絵柄に入れて下さい。
各Panelのフォーマット。このフォーマットを元に{page_panel}つのPanelを作って下さい。
- Panel layout:
 - Panel
  - Panel composition: このPanelにピッタリなショットを、フルショット、ワイドショット、ミディアムショット、クローズアップショットから選んで下さい。
  - 演出指示:. このPanelにピッタリ合った演出表示を正確に記述して下さい。
  - テキスト要素:
    - ナレーション: このPanelにぴったり合うナレーションを生成して下さい。
    - 効果音・書き文字: このPanelのシーンに最適な効果音・書き文字を生成して下さい。
  - キャラクター表情と動き:
    - 主人公: このPanelのシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: もし他の登場人物がいたら、表示して下さい。
    """

    img_up.save(img_up_path)
    image_base64 = encode_image(img_up)

    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "GOOGLE_API":
        gemini.configure(api_key=apikey) #os.getenv("gemini_key"))
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, plot_prompt])
        result = response.text
        print(result)

    elif llm_model == "OPENAI_API":
        openai_client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = openai_client.chat.completions.create(
            model=llms[llm_model],
            messages = [
                {'role': 'system',
                    'content': "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                            'text': plot_prompt},
                        {'type': 'image_url',
                            'image_url': {'url': image_base64}},
                    ]
                },
            ],
            temperature = 0.9,
            max_tokens = 2048, #256,
        )
        result = response.choices[0].message.content
        print(result)

    prompt_out = f"""
# Prerequisites:
Title: If the page plot 「{page_plot}」 includes "Title" then set it as the page title. If "Title" is not included do not set the title for this page.
Artist Requirements: {page_styles[default_style]}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose and Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] and reflecting the world of {page_styles[default_style]}
Please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
  - 主人公:
    - 名前: {chara_name}
    - Please draw the protagonist precisely with reference to the attached file `<file:{img_up_path}>`
    - 年齢、性別: Please estimate the protagonist's age and gender based on the attached file `<file:{img_up_path}>`
    - 髪型: {chara_name}は、{charas[chara_name][3]}色の髪の毛をしています。
    - 風貌: {charas[chara_name][2]}ような風貌です。
    - 服装: {chara_name}の服装は、そのシーンに合ったものです。
    - 表情: {chara_name}の表情は、そのシーンに合ったものです。

- Overall setting:
    - Canvas size: {default_size}
    - Art style: {page_styles[default_style]} (used consistently in every panel)
    - quality: {default_quality}
    - Image quality: crisp and clear (used consistently in every panel)
    - Font: Noto Sans JP (used consistently in every panel)
    - Panel Margins: Each panel should have a uniform margin of 10px on all four sides internally (between the artwork and the panel border).
    - Panel Border: All panels should have a consistent border, for example, a 2px solid black line.
    - Gutter Width: The space (gutter) between panels should be uniform, for example, 20px horizontally and vertically.
    - Page Margins: The entire page (canvas) should have a uniform margin of 30px around the area where panels are laid out.

# supplement:
- Reconfirmation of instructions is not required.
- Self-evaluation is not required.
- Please output images only.

- Page generation:
    - Page title: If the page plot includes "Title" then set it as the page title.
    - Please generate one page image exactly following the page and panel layouts based on [##Page layout] with your best effort.

## Page layout:

""" + result

    #imagefile = main(prompt_out)

    print("== Main Generation ==\n Starting Anime image generation!\n")

    source_image = open(img_up_path, "rb")
    #image_base64 = encode_image(source_image) # open(img_up, "rb")
    if llm_model == "GOOGLE_API":
        generate_model = llms[llm_model]
        gemini.configure(api_key=apikey)
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048}
        gemini_client = gemini.GenerativeModel(generate_model, generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, prompt_out])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
        generate_model = "gpt-image-1"
        client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = client.images.edit(
            model  = generate_model, #llms[llm_model], #"gpt-image-1",
            image  = source_image,
            prompt = prompt_out,
            quality= default_quality,
            size   = default_size,
            #n = generate_pages
        )

        image_response = response.data[0].b64_json
        filename   = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        imagefile  = results_path+filename+'_image.jpg'
        promptfile = results_path+filename+'_prompt.txt'
        gradio_path='./gradio_api/file='
        with open(results_path+filename+'_prompt.txt', 'a', encoding='utf-8') as f:
                f.write(prompt_out)
                #json.dump(anime_prompt, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',',': '))
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
            #image = Image.open(BytesIO(part.inline_data.data))
            #image_path = f"results/edit_{image_count}.jpg"
            #image.save(image_path)
        print(imagefile+" created!")

        gr.set_static_paths(paths=[Path.cwd().absolute()/"results"])

        sns_link= f'<a href="https://x.com/intent/post?text=100日でアイアンマンになる物語%20アイアンリーマン%20シンジの場合%20残り96日%20https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F&url={gradio_path}{imagefile}&hashtags=100日でアイアンマンになる物語,100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>' #<img src={sns_image}></a>')
        prompt_link = f'<a href="{gradio_path}{promptfile}">Prompt</a>'
        output_link = sns_link + ' | ' + prompt_link
        print(output_link)

        return imagefile, output_link
    

def camera_nodetect(image):
    image.save(img_up_path)
    return "please put image chara info: "

def encode_image(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    base64_image = f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    return base64_image

def camera_detect(image,llm_model):
    llm_model=default_model
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(llm_model+"_KEY")

    camera_prompt = "提供された画像の中に写っている人物の、おおよその年齢、性別を類推して下さい。髪型、表情、服装を詳細に簡潔に説明して下さい。"

    image.save(img_up_path)

    image_base64 = encode_image(image)
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    if llm_model == "GOOGLE_API":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llms[llm_model],generation_config=generation_config)
        response = gemini_client.generate_content([image_base64, camera_prompt])
        result = response.text
        print(result)
        return result

    elif llm_model == "OPENAI_API":
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
            max_tokens = 2048, #256,
        )
        result = response.choices[0].message.content
        print(result)

        return result

#img_up = gr.Image(label="Book Photo", sources="webcam",webcam_constraints=["rear"], 
#                  type="pil", mirror_webcam=False, width=350,height=350)

def flip(im):
    return np.fliplr(im)

def chara_picture(chara_name):
    #return f"Welcome to Gradio, {name}!"
    return charas[chara_name][1]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tab("Generate"):
                #with gr.Row():
                    img_up = gr.Image(label="Chara Photo", sources="upload",
                        type="pil", mirror_webcam=False, value=charas[default_chara][1], width=250,height=250)
                                        
                    chara_name= gr.Dropdown(choices=charas, label="Chara", value=default_chara, interactive=True, scale=1) #Textbox(label="Chara Name", interactive=True)
                    chara_name.change(chara_picture, chara_name, img_up)
                    page_plot = gr.Textbox(label="Plot", interactive=True, scale=2)

                    #prompt_out= gr.Textbox(label="Prompt", max_lines=100, placeholder="Upload photo & plot, and edit results", interactive=True, scale=2)
                    output_image= gr.Image(label="AniMaker Result")
                    output_link = gr.Markdown()
                    gr.Interface(fn=plot_image_generate, #camera_nodetect, #camera_detect,
                        inputs=[img_up,chara_name,page_plot], outputs=[output_image,output_link], #live=True, 
                        flagging_mode="never", clear_btn=None)

            with gr.Tab("New"):
                #with gr.Row():
                    new_up = gr.Image(label="New Photo", sources="upload",
                        type="pil", mirror_webcam=False, width=250,height=250,) # value=charas[default_chara][1]0)
                                        
                    new_name  = gr.Textbox(label="New member name", value="New", interactive=True, scale=1)
                    page_title= gr.Textbox(label="Title", interactive=True, scale=2)

                    output_image= gr.Image(label="AniMaker Result")
                    output_link = gr.Markdown()
                    gr.Interface(fn=plot_image_generate, #camera_nodetect, #camera_detect,
                        inputs=[new_up,new_name,page_title], outputs=[output_image,output_link], #live=True, 
                        flagging_mode="never", clear_btn=None)
                    
            with gr.Tab("Plot"):
                with gr.Row():
                    img_up = gr.Image(label="Chara Photo", sources="upload",
                        type="pil", mirror_webcam=False, value=charas[default_chara][1], width=250,height=250)
                                        
                    chara_name= gr.Dropdown(choices=charas, label="Chara", value=default_chara, interactive=True, scale=1) #Textbox(label="Chara Name", interactive=True)
                    chara_name.change(chara_picture, chara_name, img_up)
                    #page_title= gr.Textbox(label="Title", interactive=True)
                    page_plot = gr.Textbox(label="Plot", interactive=True, scale=2)

                    prompt_out= gr.Textbox(label="Prompt", max_lines=100, placeholder="Upload photo & plot, and edit results", interactive=True, scale=2)
                    gr.Interface(fn=plot_generate, #camera_nodetect, #camera_detect,
                        inputs=[img_up, chara_name,page_plot], outputs=prompt_out, #live=True, 
                        flagging_mode="never", clear_btn=None)
                    
                anime_btn = gr.Button("AniMake!")
                output_image = gr.Image(label="AniMaker Result")
                output_link = gr.Markdown()
                anime_btn.click(fn=main, inputs=prompt_out, outputs=[output_image,output_link], api_name="animaker")

            """
            with gr.Tab("Manual"):
                with gr.Row():
                    chara_name= gr.Textbox(label="Chara Name", interactive=True)
                    page_title= gr.Textbox(label="Title", interactive=True)
                    chara_out = gr.Textbox(label="Charactor",placeholder="Upload photo and edit results",interactive=True)
                    camera    = gr.Interface(fn=camera_nodetect, #camera_detect,
                        inputs=[img_up], outputs=chara_out, live=True, 
                        flagging_mode="never", clear_btn=None)
                output = gr.Markdown("Panel layout:")

                for panel in range(4):
                    with gr.Row(panel):
                        panel_shot= gr.Dropdown(choices=panel_shots,label="Panel "+str(panel+1)+" Shot", interactive=True)
                        panel_comp= gr.Textbox(label="Composition", interactive=True)
                        panel_naration = gr.Textbox(label="Naration", interactive=True)
                        #panel1_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                        #panel1_face  = gr.Textbox(label="Face", interactive=True)
                        gr.Textbox(label="Other charactors", elem_id="Other_"+str(panel), interactive=True)
                
                with gr.Row():
                    #panel1_size= gr.Dropdown(choices=panel_sizes,label="Panel1 Size", interactive=True)
                    panel1_shot= gr.Dropdown(choices=panel_shots,label="Panel 1 Shot", interactive=True)
                    panel1_comp= gr.Textbox(label="Composition", interactive=True)
                    panel1_naration = gr.Textbox(label="Naration", interactive=True)
                    #panel1_onomatope= gr.Textbox(label="Onomatope", interactive=True)
                    #panel1_face  = gr.Textbox(label="Face", interactive=True)
                    panel1_others= gr.Textbox(label="Other charactors", interactive=True)                
                
                search_btn = gr.Button("AniMake!")
                output_image= gr.Image()
                output_link = gr.Markdown()
                search_btn.click(fn=main, inputs=prompt_out, outputs=[output_image,output_link], api_name="animaker")

            
            with gr.Tab("Camera"):
                with gr.Row():
                    page_title= gr.Textbox(label="Title", interactive=True)
                    chara_out = gr.Textbox(label="Charactor",placeholder="Upload photo and edit results",interactive=True)
                    camera    = gr.Interface(fn=camera_detect,
                        inputs=[img_up], outputs=chara_out, live=True, 
                        flagging_mode="never", clear_btn=None)
            """

        with gr.Column():
            with gr.Accordion(label="Anime Options:", open=False):
                page_style= gr.Dropdown(choices=page_styles,label="Anime Style", value=default_style, interactive=True)
                page_size = gr.Dropdown(choices=page_sizes,label="Canvas size",  value=default_size, interactive=True)
                page_story= gr.Dropdown(choices=page_storys,label="Generate/Manual", value=default_story, interactive=True)
                image_quality= gr.Dropdown(choices=image_qualities,label="Image quality", value=default_quality, interactive=True)
                generate_page= gr.Dropdown(choices=generate_pages,label="Generate page/s",value=default_page, interactive=True)
                page_panel  = gr.Dropdown(choices=page_panels,label="# of panels in a page",value=default_panel, interactive=True)

                llm_model= gr.Dropdown(choices=llms,label="LLM", interactive=True, value=default_model)
                llm_key  = gr.Textbox(label="LLM API Key", interactive=True,) #value=default_key,placeholder="Paste your LLM API key here",)
                llm_usage= gr.Markdown("https://platform.openai.com/usage")
                num_steps= gr.Slider(minimum=1,maximum=20,value=default_steps,step=1, label="Steps",interactive=True)
            with gr.Accordion(label="Charactors:", open=False):
                for chara in charas:
                    with gr.Row(chara):
                        chara_image= gr.Image(charas[chara][1], label=chara)
                        chara_chara= gr.Markdown(charas[chara][2])
                with gr.Row():
                    chara_name = gr.Textbox(label="Add Chara Name", interactive=True)
                    chara_desc = gr.Textbox(label="Chara Description", placeholder="Add photo and chara description", interactive=True)
                    chara_color= gr.Dropdown(choices=colors, label="Chara Color", value=default_color, interactive=True)
                    chara_up = gr.Image(label="Person's photo", sources="upload",
                        type="pil", mirror_webcam=False, width=200,height=200)
                    chara_btn = gr.Button("Add Anime Chara")
                    chara_btn.click(fn=add_chara, inputs=[chara_up,chara_name,chara_desc,chara_color],
                        outputs=[gr.Image(label="Generated Anime Chara",width=200,height=200)], api_name="addchara")
    """
    [gr.Image(
        label="ANIME Result:",
    #gr.Gallery(
        #show_label=True,
        #elem_id="Gallery",
        #columns=[2],
        #object_fit="contain",
        #height="auto",
    ),
    #gr.Textbox(label="Error Messages"),
    ],
    """
    #sns_image = image_path+"sns3.jpg" 
    #twitter_mark= gr.Markdown(f'<a href="https://x.com/intent/post?text=100日でアイアンマンになる物語%20残り◯日%20&url=https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F&hashtags=ironman100&openExternalBrowser=1">Post SNS</a>') #<img src={sns_image}></a>')
    with gr.Accordion(label="Anime Gallery:", open=False):
        gr.Gallery(ret_data(5), columns=6, show_label=True, show_download_button=True, show_share_button=True, allow_preview=True)

parser = argparse.ArgumentParser(description="Gradio UI for Anime Maker")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
args = parser.parse_args()

demo.launch(server_name=args.ip,server_port=args.port, auth=("usr","pswd1"))
