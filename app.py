import gradio as gr
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_anthropic import ChatAnthropic
#from browser_use import Agent
import asyncio
import os, io, requests, json
import base64
from io import BytesIO
from PIL import Image,ImageFilter
import numpy as np
import argparse

from datetime import datetime
from dateutil.parser import parse

from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
#from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import google.generativeai as gemini
from google import genai
from google.genai import types
#from google.cloud import vision
#from google.oauth2 import service_account

page_styles= {"Jp 90s": "日本の90年代アニメ風。セル画のような色使いと質感で、太い輪郭線、大きな瞳、光沢のある髪のアニメスタイル",
        "Arukikata": "地球の歩き方風の表紙",
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

llms = {"OPENAI_API": "gpt-4o-mini", #"gpt-image-1",
        "GOOGLE_API": "gemini-2.0-flash-exp", #"gemini-2.0-flash-preview-image-generation",
        "ANTHOLOPIC": "claude-3-5-sonnet-latest"}
options = {"""
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
"""}

"""
cover_anime = {"cover":[1],
               "page":[1,2,3,4]}
page_choices= []
for k,v in cover_anime.items():
    for v2 in v:
        page_choices.append(k.title()+str(v2))

cover_pages_choices = {"Cover": 0,
                       "Page 1": 1,
                       "Page 2": 2,
                       "Page 3": 3,
                       "Page 4": 4,
}
"""

DEF_LLM = "GOOGLE_API" #"OPENAI_API" #
default_key  = "api_key ..."
default_style= "Jp 90s"
default_story= "Generate"
default_size = "1024x1536" #"1536x1024" #"1024x1024"
default_quality= "high"
default_page   = 1
default_panel  = 5
default_color  = "指定なし"


default_steps= 10
default_cat  = "Book"

#result_folder= "./results"
#up_folder    = "./up"
results_path = './results/'
img_up_path= './image/img_up.jpg'
chara_path = "./chara/"
#chara_path = ""

gradio_path='./gradio_api/file='
gr.set_static_paths(paths=[Path.cwd().absolute()/"results"])
story_name = "100日でアイアンマンになる物語"

default_chara= "シンジ"
charas = {
    "シンジ": ["Shinji", chara_path+"shinji_anime.jpg","赤いトライアスロンウェアに身を包んだ、30代中肉中背の男性。髪は短く、黒色の髪。ニヤリと笑っている","黒"],
    "ノリコ": ["Noriko", chara_path+"noriko_anime.jpg","ブルーのランニングウェアに身を包んだ、ギャルっぽい女の子。髪はショートで青色の髪。スポーティーで笑顔が可愛い","青"],
    "ゴトウ": ["Goto",   chara_path+"goto_anime.jpg","黒いウェアにサングラスをかけている。40代の男性で鬼軍曹のような厳しい雰囲気。髪はベリーショートで、真っ黒","黒"],
    "ハラダ": ["Harada", chara_path+"harada_anime.jpg","オレンジのトライアスロンウェアを着た、40代の若手経営者。髪はお洒落なパーマで金髪。キラキラして自信ありげな笑みを浮かべる","金"],
    "マツイ": ["Matsui", chara_path+"matsui_anime.jpg","白いランニングウェアを着たマッチョな30代男性。寡黙だが子煩悩なパパでもある。髪の毛はミディアムで緑色","紫"],
    "ケン":   ["Ken",   chara_path+"ken_anime.jpg","黒のトライアスロンウェアを着た痩せぎすな30代男性。自信なさげだが内なる闘志を秘めている。髪はサラサラで赤色","赤"],
    "New":   ["new",   chara_path+"others_anime.jpg","それぞれメンバーの仲間たち","黒"],
#    "ナオト": ["Naoto",  chara_path+"naoto_anime.jpg","ナオトは世界中を旅している20代の大学生。背は低いが、足が速く、引き締まった体をしている","青"],
#    "ケニー":  ["Kenny", chara_path+"kenny_anime.jpg","ケニーは世界中を旅している20代の大学生。引き締まった体をしている","赤"],
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

def add_chara(LLM, chara_up,chara_name,chara_desc,chara_color):
    print("== Chara Generation ==\n Starting {chara_name} image generation!\n")
    apikey = os.getenv(LLM+"_KEY")

    chara_up.save(chara_path)
    source_image = open(chara_path+chara_name+".jpg", "rb")

    image_base64 = encode_image(chara_up) # open(img_up, "rb")

    anime_prompt= f"""ここに写っている人物を、日本の90年代アニメ風の画像に変換して下さい。
        セル画のような色使いと質感で、太い輪郭線、光沢のある髪のアニメスタイルです。画像のみを出力して下さい。
        顔や表情は写真に正確に忠実に表現して下さい。
        この人物のイメージは、{chara_desc}で、髪の毛は{chara_color}色で印象的な髪型に変更して下さい。服装も全体的に{chara_color}色風にして下さい。
        ロゴや文字などは架空のものに変更して下さい。
        背景は海沿いの青空にして下さい。"""
    
    charafile = generate_image(LLM,apikey, anime_prompt,source_image)
    return charafile

"""
async def main(LLM, prompt_out): #img_up, llm_model, prompt_out):
    #llm_model=default_model
    print(f"== Main Generation ==\n Starting Anime image generation by {LLM}!\n")
    #print(llm_model)
    #print(prompt_out)
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(LLM+"_KEY")
    #source_image = open(img_up_path, "rb")
    #image_base64 = encode_image(source_image) # open(img_up, "rb")
    #img = Image.open(filename)
    with open(img_up_path, 'rb') as f:
        data = f.read()
    source_image = Image.open(BytesIO(data))

    filename   = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    #imagefile = results_path+filename+'_image.jpg'
    promptfile = results_path+filename+'_prompt.txt'

    if LLM == "GOOGLE_API":
        generate_model = llms[LLM]
        #gemini.configure(api_key=apikey)
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 4098}
        #gemini_client = gemini.GenerativeModel(generate_model, generation_config=generation_config)
        client  = genai.Client(api_key=apikey)
        #response = gemini_client.generate_content([image_base64, prompt_out])
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation", #models/gemini-2.5-pro-preview-05-06", # gemini-2.0-flash-exp",
            contents=[prompt_out, source_image],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )
        image_count = 0
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                with open(results_path+filename+'_part.txt', "a", encoding="utf-8") as f:
                    f.write(part.text + "\n\n")
                print(part.text)
            elif part.inline_data is not None:
                image_count += 1
                with open(results_path+filename+'_gemprompt'+str(image_count)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(prompt_out)
                image = Image.open(BytesIO(part.inline_data.data))
                #edited_image.save("edited_image.jpg")
                imagefile = results_path+filename+'_gemimage'+str(image_count)+'.jpg'
                image.save(imagefile)

            #with open(imagefile, "wb") as f:
                #f.write(base64.b64decode(image_response))

        #result = response.text
        #return result

    elif LLM == "OPENAI_API":
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
        imagefile = results_path+filename+'_gptimage.jpg'
        with open(promptfile, 'a', encoding='utf-8') as f:
                f.write(prompt_out)
                #json.dump(anime_prompt, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',',': '))
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
            #image = Image.open(BytesIO(part.inline_data.data))
            #image_path = f"results/edit_{image_count}.jpg"
            #image.save(image_path)

    sns_link= f"<a href="https://x.com/intent/post?text={story_name}%20
        アイアンリーマン%20シンジの場合%20残り96日%20
        https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F
        &url={gradio_path}{imagefile}
        &hashtags={story_name},100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>
        "
    prompt_link = f'<a href="{gradio_path}{promptfile}">Prompt</a>'
    output_link = sns_link + ' | ' + prompt_link
    print(imagefile+" created!")

    return imagefile, output_link
"""

def plot_generate(LLM, chara_name,page_plot, title_plot="plot", cover_pages=1):
    print(f"== Prompt Generation ==\n Starting {title_plot} for {cover_pages} creation by {LLM} for {chara_name}!\n")
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(LLM+"_KEY")
    openai_client = OpenAI(api_key=apikey)

    charas_prompt= ""
    if title_plot in ["title","cover"]:
        system_content = "このシステムは、テキストが提供された時に、それをタイトルとした物語のあらすじを300字程度で、起承転結を付けて生成します。"
        use_plot = generate_text(LLM,apikey, system_content, page_plot)
    else:
        use_plot = page_plot
        for chara in charas:
            charas_prompt += f"""- 登場人物「{chara}」: {chara}の顔、表情、年齢、性別は、この画像 `<file:{charas[chara][1]}>` を正確に反映して下さい。
            この人物の特徴は、{charas[chara][2]}で、{charas[chara][3]}色の髪型をしています。
            """
    print(title_plot+" = " + use_plot)

    common_prompt = f"""# Prerequisites:
Artist Requirements: {page_style}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose and Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
    {charas_prompt}
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
    """

    plot_prompt = f"""「{chara_name}」はこの話の主人公です。他の登場人物としては、{charas.keys()}がいます。
各登場人物の特徴は以下のようなものです。各登場人物の服装と表情は、そのシーンに合ったものにして下さい。
{charas_prompt}
各登場人物は、そのシーンに合った服装、表情をしています。
以下の[# プロット]から、n=1からn={generate_pages[3]}までの、{generate_pages[3]}ページのプロンプトを生成して下さい。1ページには、m=1からm={page_panels[4]}までの{page_panels[4]}つのPanelを作り、各Panel毎に以下のフォーマットに従って、1ページに{page_panels[4]}つのpanelを記述したプロンプトを作成してください。
プロット全体を通した[# Text Name Title] をつけて下さい。
各ページ毎の[# ページタイトル]もつけて下さい。
プロットに記述したランドマークや商品、人物はなるべく正確に写実的に表現して絵柄に入れて下さい。
各Panelのフォーマットはこのフォーマットを元に、1ページに{page_panels[4]}つのPanelを作って下さい。

■ ■ テキストネーム ■ ■
# プロット:
{use_plot}
■ ■ [# Text Name Title] ■ ■
# Page n : [# ページタイトル]
## Panels layout:
### Panel m
  — Panel composition: このPanelにピッタリなショットを、フルショット、ワイドショット、ミディアムショット、クローズアップショットから選んで下さい。
  — 演出指示:. このPanelにピッタリ合った演出表示を正確に記述して下さい。
  — テキスト要素:
    - ナレーション: このPanelにぴったり合うナレーションを生成して下さい。
    - 効果音・書き文字: このPanelのシーンに最適な効果音・書き文字を生成して下さい。
  — キャラクター表情と動き:
    - 主人公: このPanelのシーンに最適な主人公の表情を類推し表示して下さい。
    - その他の登場人物: もし他の登場人物がいたら、表示して下さい。
    """

    if title_plot == "cover":
        direction = "\n ## Please generate one page cover with one big image exactly following the prompt with your best effort. \n"
        generated_prompt = direction + f""" Please generate Manga Cover page.
            漫画のカバーを{cover_style}のように作って下さい。
            {page_plot}をタイトルとして、画像内の上側に大きく配置。
            著者名として、「アニメイカー by {llms[LLM][:6].upper()}」を題名の下に表示。
            写真を大きく中心に、印象的に配置。
            全体感はプロット{use_plot}のテイストを入れて描写。
            """
    else:
        system_content  = "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"
        direction = f"\n ## Please generate one page image exactly following the page and panel layouts for [# Page {cover_pages}] with your best effort. \n"
        generated_prompt = direction + generate_text(LLM,apikey, system_content, plot_prompt)

    anime_prompt = common_prompt + generated_prompt
    
    return use_plot, anime_prompt

async def image_generate(LLM, prompt_out):
    #llm_model=default_model
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(LLM+"_KEY")

    print(f"== Main Generation ==\n Starting Anime image generation by {LLM}!\n")

    img_up.save(img_up_path)
    #image_base64 = encode_image(img_up)

    #imagefile = main(prompt_out)

    source_image = open(img_up_path, "rb")
    imagefile, output_link = generate_image(LLM,apikey, prompt_out,source_image)

    return imagefile, output_link


async def plot_image_generate(LLM, img_up,chara_name,page_plot, cover_pages=1):
    print(f"== Prompt Image Generation ==\n {cover_pages} image by {LLM} for {chara_name}!\n")
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(LLM+"_KEY")
    
    if cover_pages == 0:
        use_plot, prompt_out = plot_generate(LLM, chara_name,page_plot, "cover", cover_pages)
    else:
        if len(page_plot) < 50:
            use_plot, prompt_out = plot_generate(LLM, chara_name,page_plot, "title", cover_pages)
        else:
            use_plot, prompt_out = plot_generate(LLM, chara_name,page_plot, "plot", cover_pages)

    img_up.save(img_up_path)
    #image_base64 = encode_image(img_up)

    #imagefile = main(prompt_out)
    print(f"== Image Generation ==\n Starting Anime image generation by {LLM}!\n")

    source_image = open(img_up_path, "rb")
    #image_base64 = encode_image(source_image) # open(img_up, "rb")
    #with open(img_up_path, 'rb') as f:
        #data = f.read()
    #source_image = Image.open(BytesIO(data))

    #filename   = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    #imagefile = results_path+filename+'_image.jpg'
    #promptfile = results_path+filename+'_prompt.txt'

    imagefile,promptfile = generate_image(LLM,apikey, prompt_out,source_image)

    sns_link= f"""<a href="https://x.com/intent/post?text={story_name}%20
        アイアンリーマン%20シンジの場合%20残り96日%20
        https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F
        &url={gradio_path}{imagefile}
        &hashtags={story_name},100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>
        """
    prompt_link = f'<a href="{gradio_path}{promptfile}">Prompt</a>'
    output_link = sns_link + ' | ' + prompt_link

    return use_plot, imagefile, output_link


def camera_nodetect(image):
    image.save(img_up_path)
    return "please put image chara info: "

def encode_image(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    base64_image = f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    return base64_image

def camera_detect(image,LLM):
    #llm_model=default_model
    #if llm_key:
    #    apikey = llm_key
    #else:
    apikey = os.getenv(LLM+"_KEY")

    camera_prompt = "提供された画像の中に写っている人物の、おおよその年齢、性別を類推して下さい。髪型、表情、服装を詳細に簡潔に説明して下さい。"

    image.save(img_up_path)

    image_base64 = encode_image(image)
    if LLM == "GOOGLE_API":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llms[LLM],generation_config=ai_gen_config)
        response = gemini_client.generate_content([image_base64, camera_prompt])
        result = response.text
        print(result)
        return result

    elif LLM == "OPENAI_API":
        openai_client = OpenAI(api_key=apikey)
        #messages = create_message(SYSTEM_ROLE_CONTENT, PROMPT_TEMPLATE, image_base64)
        response = openai_client.chat.completions.create(
            model=llms[LLM],
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
            max_tokens = 4098, #2048, #256,
        )
        result = response.choices[0].message.content
        print(result)

        return result

#img_up = gr.Image(label="Book Photo", sources="webcam",webcam_constraints=["rear"], 
#                  type="pil", mirror_webcam=False, width=350,height=350)

def flip(im):
    return np.fliplr(im)
def chara_picture(chara_name):
    return charas[chara_name][1]
def llm_change(LLM):
    return llms[LLM]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Accordion(label="AniMaker!", open=False):
            LLM = gr.Dropdown(choices=llms,label="0. LLM", interactive=True, value=DEF_LLM)
            llm_model = gr.Textbox(label="0. LLM Model", value=llms[DEF_LLM], interactive=True)
            LLM.change(llm_change, LLM, llm_model)
            llm_key = gr.Textbox(label="0. LLM API Key", interactive=True, value=default_key) #,placeholder="Paste your LLM API key here",)
            animaker_usage = gr.Markdown(f"""
                # How to use AniMaker:
                ## 0. LLMとAPI Keyをセット
                ## New Story:
                1. 写真をアップして
                2. 適当なタイトル(例: シンガポールで大冒険)を入れて
                3. アニメ風、アメコミ風などを選べば
                4. アニメの表紙と4ページのマンガがAniMaker!
                ## Chara Plot:
                0. キャラを事前登録
                1. キャラを選んで
                2. あらすじを入れると
                3. 4ページ分のプロンプト生成
                4. それを微修正して、5コマのマンガがAniMaker!
                ### AI Usages:
                - https://platform.openai.com/usage
                - https://gemini.google.com/usage
                """)
    with gr.Row():
        with gr.Column():
            with gr.Tab("StoryMaker"):
                new_up = gr.Image(label="1. Upload Photo", sources="upload",
                    type="pil", mirror_webcam=False, width=250,height=250,) # value=charas[default_chara][1]0)
                #new_name  = gr.Textbox(label="New member name", value="New", interactive=True, scale=1)
                page_title = gr.Textbox(label="2. Title", placeholder="Just put title (e.g. Singapore trip...) for anime", interactive=True, scale=2)
                cover_pages= gr.Dropdown(choices=[0,1,2,3,4], label="3. Generate Cover (0) or Pages (1 - 4) for Anime", interactive=True)
                cover_style= gr.Dropdown(choices=page_styles,label="3. Cover Image Style", value=default_style, interactive=True)

                new_btn    = gr.Button("4. AniMaker!")
                #with gr.Accordion(open=False):
                output_image = gr.Image(label="AniMaker Image")
                output_link  = gr.Markdown()
                use_plot = gr.Markdown(label="Generated plot and image for AniMaker")
                new_btn.click(fn=plot_image_generate, inputs=[LLM, new_up,cover_style,page_title, cover_pages], 
                    outputs=[use_plot,output_image,output_link], api_name="new_generate")
            """
            with gr.Tab("PlotMaker"):
                #with gr.Row():
                    img_up = gr.Image(label="Chara Photo", sources="upload",
                        type="pil", mirror_webcam=False, value=charas[default_chara][1], width=250,height=250)
                    chara_name= gr.Dropdown(choices=charas, label="Chara", value=default_chara, interactive=True, scale=1) #Textbox(label="Chara Name", interactive=True)
                    chara_name.change(chara_picture, chara_name, img_up)
                    page_plot = gr.Textbox(label="Plot", interactive=True, scale=2)

                    #prompt_out= gr.Textbox(label="Prompt", max_lines=100, placeholder="Upload photo & plot, and edit results", interactive=True, scale=2)
                    gen_btn = gr.Button("AniMaker!")
                    #with gr.Accordion(open=False):
                    use_plot = gr.Markdown(label="Generated plot and image for AniMaker")
                    output_image= gr.Image(label="AniMaker Image")
                    output_link = gr.Markdown()
                    gen_btn.click(fn=plot_image_generate, inputs=[LLM, img_up,chara_name,page_plot], 
                        outputs=[use_plot,output_image,output_link], api_name="plot_image_generate")
            """
            with gr.Tab("PlotMaker"):
                #with gr.Row():
                img_up = gr.Image(label="1. Chara Photo", sources="upload",
                    type="pil", mirror_webcam=False, value=charas[default_chara][1], width=250,height=250)         
                chara_name= gr.Dropdown(choices=charas, label="1. Chara", value=default_chara, interactive=True, scale=1) #Textbox(label="Chara Name", interactive=True)
                chara_name.change(chara_picture, chara_name, img_up)
                page_plot = gr.Textbox(label="2. Plot", placeholder="Put your story plot here", interactive=True, scale=2)
                plot_btn = gr.Button("3. Generate Prompt")
                prompt_out= gr.Textbox(label="3. Prompt Out", max_lines=500, 
                    placeholder="Upload photo & plot, then edit results", interactive=True, scale=2)
                plot_btn.click(fn=plot_generate, inputs=[LLM, chara_name,page_plot], outputs=[page_plot, prompt_out], api_name="plot_generate")
                    
                anime_btn = gr.Button("4. AniMaker!")
                output_image = gr.Image(label="AniMaker Image")
                output_link = gr.Markdown()
                anime_btn.click(fn=image_generate, inputs=[LLM, prompt_out], 
                    outputs=[use_plot,output_image,output_link], api_name="animaker")
            """
            with gr.Tab("Camera"):
                with gr.Row():
                    page_title= gr.Textbox(label="Title", interactive=True)
                    chara_out = gr.Textbox(label="Charactor",placeholder="Upload photo and edit results",interactive=True)
                    camera    = gr.Interface(fn=camera_detect,
                        inputs=[img_up], outputs=chara_out, live=True, 
                        flagging_mode="never", clear_btn=None)
            """
        with gr.Column():
            with gr.Accordion(label="Options:", open=False):
                page_style= gr.Dropdown(choices=page_styles,label="Anime Style", value=default_style, interactive=True)
                page_size = gr.Dropdown(choices=page_sizes,label="Canvas size",  value=default_size, interactive=True)
                page_story= gr.Dropdown(choices=page_storys,label="Generate/Manual", value=default_story, interactive=True)
                image_quality= gr.Dropdown(choices=image_qualities,label="Image quality", value=default_quality, interactive=True)
                generate_page= gr.Dropdown(choices=generate_pages,label="Generate page/s",value=default_page, interactive=True)
                page_panel  = gr.Dropdown(choices=page_panels,label="# of panels in a page",value=default_panel, interactive=True)

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
                    chara_btn.click(fn=add_chara, inputs=[LLM, chara_up,chara_name,chara_desc,chara_color],
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
    with gr.Row():
        with gr.Accordion(label="Anime Gallery:", open=False):
            gr.Gallery(ret_data(5), columns=6, show_label=True, show_download_button=True, show_share_button=True, allow_preview=True)


"""
with open(img_up_path, 'rb') as f:
    data = f.read()
source_image = Image.open(BytesIO(data))
filename   = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#imagefile = results_path+filename+'_image.jpg'
promptfile = results_path+filename+'_prompt.txt'
"""

ai_gen_config = {"temperature":0.9, 
                 "top_p":0.95, "top_k":40, 
                 "max_output_tokens": 4098} #8192, 2048, #256,

def generate_text(LLM, apikey, system_content, in_prompt):
    if apikey is None:
        apikey = os.getenv(LLM+"_KEY")
    llm_model= llms[LLM]

    if LLM == "GOOGLE_API":
        gemini.configure(api_key=apikey)
        gemini_client = gemini.GenerativeModel(llm_model, 
            generation_config=ai_gen_config) #{"temperature":temp_config, "max_output_tokens":max_config})
        response= gemini_client.generate_content(in_prompt) #[image_base64, plot_prompt])
        result  = response.text
        
    elif LLM == "OPENAI_API":
        gpt_client = OpenAI(api_key=apikey)
        #system_content = "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"
        response = gpt_client.chat.completions.create(
            model=llm_model,
            messages = [
                {'role': 'system',
                    'content': system_content},
                {'role': 'user',
                    'content': [
                        {'type': 'text',
                            'text': in_prompt},
                        #{'type': 'image_url',
                            #'image_url': {'url': image_base64}},
                    ]
                },
            ],
            #generation_config = ai_gen_config
            #temperature = temp_config,
            #max_tokens = max_config,
        )
        result = response.choices[0].message.content

    #print("Generated Plot: "+result)
    return result
    
def generate_image(LLM,apikey, in_prompt,source_image):
    if apikey is None:
        apikey = os.getenv(LLM+"_KEY")
    llm_model= llms[LLM]
    
    #source_image = open(img_up_path, "rb")
    #image_base64 = encode_image(source_image) # open(img_up, "rb")
    #img = Image.open(filename)
    
    with open(img_up_path, 'rb') as f:
        data = f.read()
    #source_image = Image.open(BytesIO(data))
    image_base64 = encode_image(Image.open(BytesIO(data)))

    filename = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    promptfile= results_path+filename+'_prompt.txt'
    imagefile = results_path+filename+'_image.jpg'

    if LLM == "GOOGLE_API":
        #gemini.configure(api_key=apikey)
        client  = genai.Client(api_key=apikey)
        #gemini_client = gemini.GenerativeModel(llm_model, 
            #generation_config={"temperature":temp_config, "max_output_tokens":max_config})
        response = client.models.generate_content(model=llm_model,
            contents=[in_prompt, image_base64],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image']))
                    #generation_config=ai_gen_config)
                    #temperature=temp_config,
                    #top_p=0.95,
                    #top_k=40,
                    #max_output_tokens=max_config)
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                with open(promptfile, "a", encoding="utf-8") as f:
                    f.write(part.text + "\n\n")
                print(part.text)
            elif part.inline_data is not None:
                #image_count += 1
                #promptfile= results_path+filename+'_prompt'+str(image_count)+'.txt'
                #imagefile = results_path+filename+'_image'+str(image_count)+'.jpg'
                with open(promptfile, 'a', encoding='utf-8') as f:
                    f.write(in_prompt)
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(imagefile)

            print("ImageFile saved: " + imagefile)
        
    elif LLM == "OPENAI_API":
        gpt_client = OpenAI(api_key=apikey)
        generate_model = "gpt-image-1"
        response = gpt_client.images.edit(
            model  = generate_model,
            image  = source_image,
            prompt = in_prompt,
            quality= default_quality,
            size   = default_size,
            #n = generate_pages
        )
        image_response = response.data[0].b64_json
        #promptfile= results_path+filename+'_prompt.txt'
        #imagefile = results_path+filename+'_image.jpg'

        with open(promptfile, 'a', encoding='utf-8') as f:
            f.write(in_prompt)
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
            #image = Image.open(BytesIO(part.inline_data.data))
            #image_path = f"results/edit_{image_count}.jpg"
            #image.save(image_path)
        print("ImageFile saved: "+imagefile)

    return imagefile,promptfile


parser = argparse.ArgumentParser(description="Gradio UI for Anime Maker")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8180, help="Port to listen on")
args = parser.parse_args()
demo.launch(server_name=args.ip,server_port=args.port, auth=("usr","pswd1"))
