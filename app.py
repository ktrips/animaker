import gradio as gr
import asyncio
import os, io, requests, json
import base64
from io import BytesIO
from PIL import Image,ImageFilter
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import argparse

from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
animaker_usr = os.getenv("ANIMAKER_USR")
animaker_pswd= os.getenv("ANIMAKER_PSWD")
DEF_LLM      = "GOOGLE_API" #"OPENAI_API"
default_key  = "" #os.getenv(DEF_LLM+"_KEY") #

from openai import OpenAI
import google.generativeai as gemini
from google import genai
from google.genai import types
#from google.genai.types import GenerateContentConfig, Modality #, ImageConfig
#from google.cloud import vision
#from google.oauth2 import service_account

page_styles= {"Jp 90s": "日本の90年代アニメ風。セル画のような色使いと質感で、太い輪郭線、大きな瞳、光沢のある髪で、クリアなタッチのアニメスタイル",
        "Arukikata": "地球の歩き方風の表紙",
        "Ukiyoe": "浮世絵の葛飾北斎風に変換。太い輪郭線と落ち着いた和風色彩で描き、背景には浮世絵スタイルの山や波を加える。",
        "Retro": "レトロゲーム風に変換。8bitドット絵スタイルで、カラフルかつレトロな雰囲気にする。",

        "Ghibli": "スタジオジブリ風のアニメイラストに変換。自然豊かな背景、柔らかい線画、温かみのある色合いを表現。",
        "Pixar": "ディズニー・ピクサー風の3Dアニメスタイルに変換。大きな瞳、立体的な質感、明るい色合いで描画。",
        "Jojo": "アニメのジョジョ風（ジョジョの奇妙な冒険）に変換。太い輪郭線、激しい陰影、ポーズを強調し、カラフルな背景でジョジョの奇妙な冒険風にする。",
        "Dragonquest": "ドラゴンクエスト風に変換。鳥山明風のイラストで、剣や魔法、明るいファンタジー世界の背景、ポップな表情で描いてください。",
        "Slumdank": "アニメのスラムダンク風に変換してください。ユニフォームを着て、汗や動きのあるポーズ、青春漫画風のリアルな線画で描いてください。",
        "Pokemon": "ポケモン風のキャラクターデザインに変換。アニメタッチでかわいらしく、明るい色合いで仕上げ。",
        "Eva": "エヴァンゲリオン風に変換。モダンなメカニカルデザインと、ダークでシリアスな色調を追加。",
        "Kimetsu": "鬼滅の刃風に変換。和装デザインと、呼吸技のエフェクトを加えたスタイリッシュなタッチにする。",
        "Akira": "アニメのAKIRA風に変換。ネオ東京の荒廃した都市を背景に、赤いバイクに乗ったキャラクターを描く。赤や黒を基調とした色使いで、ネオンライトや未来的な都市の雰囲気を追加。",
        "Chikawa": "ちいかわ風に変換してください。丸くて小さな体型、シンプルな線画、ほのぼのした背景で描いてください。",
}

page_sizes = {"1024x1024": "5:4", #"1:1",
    "1024x1536": "9:16", #"4:5",
    "512x768": "9:16",
    "1536x1024": "16:9",
    "768x512": "16:9",
    "4コマ": "9:16",
    "ポスター": "9:16"}
#default_size   = "1024x1024" #"1024x1536" #"1536x1024"  #"768x768"
image_qualities = {"low":"1K",
                "middle":"2K",
                "high":"4K",}
#default_quality= "low"
default_page   = 2
default_panel  = 5
default_color  = "指定なし"
"""
page_storys= {"Manual": "主人公を中心として、入力されたページ構成を変更せず、忠実に従って下さい。",
        "Generate": "主人公を中心としたストーリーを、ページ構成に従って、できる限り詳細に生成して下さい。",
        "Hybrid": "主人公を中心としたストーリーを、入力された情報に付加して、作り上げて下さい。"}
"""
colors = ["指定なし","黒","茶","赤","青","黄","緑","紫","ピンク","オレンジ","白"]

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part

llms = {"OPENAI_API": "gpt-4o-mini", #"gpt-image-1",
        "GOOGLE_API": "gemini-2.0-flash", #"gemini-2.0-flash-preview-image-generation",
        "ANTHOLOPIC_API": "claude-3-5-sonnet-latest"}
llms_image = {"OPENAI_API": "gpt-image-1",
        "GOOGLE_API": "gemini-3-pro-image-preview", #"gemini-2.0-flash-preview-image-generation",
        "ANTHOLOPIC_API": "claude-3-5-sonnet-latest"}

genai_config = {"temperature":0.9, 
            "top_p":0.95, "top_k":40, #0.95, 
            "max_output_tokens": 8192,}  #4098 #2048, #256,
dream_list = ["", "宇宙飛行士","アイドル","スポーツ選手","人気YouTuber","人気アナウンサー","プロゲーマー","ドクターX",
            "ノーベル賞","パティシエ","大統領","総理大臣","エベレスト登頂","ロックスター","アーティスト"]
cover_page_list = ["Cover 0","Page 1","Figure 5","ThreeV 6","GCode 7"] #"Page 2","Page 3","Page 4"
runner_flag = "無し"
title3d_flag= "NO"

default_style  = "Jp 90s"
default_steps= 10
#default_story  = "Generate"
#default_cat  = "Book"

results_path= './results/'
img_up_path = './image/img_up.jpg'
chara_path  = "./chara/"
gradio_path='./gradio_api/file='
gr.set_static_paths(paths=[Path.cwd().absolute()/"results"])

all_charas = { #"トライアスロンメンバー":["Members", chara_path+"Shinji_Noriko_Harada_Matsui_Goto.jpg", "シンジ、ノリコ、ゴトウ、ハラダ、マツイのトライアスロンのメンバー","黒"],
    "シンジ":["Shinji",chara_path+"shinji_anime.jpg","赤いトライアスロンウェアに身を包んだ、30代中肉中背の男性。髪は短く、黒色の髪。ニヤリと笑っている","黒"],
    "ノリコ":["Noriko",chara_path+"noriko_anime.jpg","ブルーのランニングウェアに身を包んだ、ギャルっぽい女の子。髪はショートで青色の髪。スポーティーで笑顔が可愛い","青"],
    "ゴトウ":["Goto",  chara_path+"goto_anime.jpg","黒いウェアにサングラスをかけている。40代の男性で鬼軍曹のような厳しい雰囲気。髪はベリーショートで、真っ黒","黒"],
    "ハラダ":["Harada",chara_path+"harada_anime.jpg","オレンジのトライアスロンウェアを着た、40代の若手経営者。髪はお洒落なパーマで金髪。キラキラして自信ありげな笑みを浮かべる","金"],
    "マツイ":["Matsui",chara_path+"matsui_anime.jpg","白いランニングウェアを着たマッチョな30代男性。寡黙だが子煩悩なパパでもある。髪の毛はミディアムで緑色","紫"],
    "ケン":  ["Ken",   chara_path+"ken_anime.jpg","黒のTシャツに蛍光グリーンのマウンテンジャケットを着たバックパッカー。髪はサラサラで赤色","赤"], 
    "ナオト":["Naoto", chara_path+"naoto_anime.jpg","ナオトは世界中を旅している20代の大学生。背は低いが、足が速く、引き締まった体をしている","青"],
    "ユラ":  ["Yura",  chara_path+"yura_anime.jpg","ユラは芯が強く賢い女子高生だが、優しくいつも笑顔でみんなを和ませている。","ピンク"],
    #"旅仲間":["Travelers",chara_path+"ken_naoto.jpg","ケンとナオトの旅仲間","黒"],
    #"New":  ["new",   chara_path+"new_anime.jpg","それぞれメンバーの仲間たち","黒"],
    }
trip_charas = {"ケン":  ["Ken",   chara_path+"ken_anime.jpg","黒のTシャツに蛍光グリーンのマウンテンジャケットを着たバックパッカー。髪はサラサラで赤色","赤"], # トライアスロンウェアを着た痩せぎすな30代男性。自信なさげだが内なる闘志を秘めている。髪はサラサラで赤色","赤"],
    "ナオト":["Naoto", chara_path+"naoto_anime.jpg","ナオトは世界中を旅している20代の大学生。背は低いが、足が速く、引き締まった体をしている","青"],}
default_name = "シンジ"
#default_chara= {default_name: default_name} #"トライアスロンメンバー" #"ケン" #"シンジ"
default_story = "Ironman" #100日でアイアンマンになる物語"
# ["0 What","1 Where", "2 Who", "3 When", "4 Why", "5 How", "6 Story",]
stories = {
    "Ironman": ["スイム3.8km、バイク180km、ラン42kmの最長距離トライアスロンのアイアンマン・レースを、",
                "日本とオーストラリア・ケアンズで、",
                "それぞれ個性を持ったシンジ、ノリコ、ハラダ、マツイ、ゴトーの5人が、",
                all_charas,
                "30代から50代のサラリーマンとして、",
                "それをやり始めるきっかけから、",
                "それを達成するまでの準備や厳しい練習を通して、",
                "アイアンマン・レースを完走するまでのスポーツ感動物語"],
    "Mideast": ["バックパッカー旅行を、",
                "イタリアからトルコ、イスラエルを経て、エジプトまで、",
                "大学院生のケンとナオトが、",
                trip_charas,
                "1998年の大学院最後の休みの40日間で、",
                "自分探しと視野を広げ、文化交流のために、",
                "海路と陸路で移動しながら、艱難辛苦と様々な人との出会いを通して、",
                "10カ国を巡る冒険旅行記"]
}
default_desc = stories[default_story][0]+stories[default_story][2]+stories[default_story][7]
"""
charas = {
    #"トライアスロンメンバー":["Members", chara_path+"Shinji_Noriko_Harada_Matsui_Goto.jpg", "シンジ、ノリコ、ゴトウ、ハラダ、マツイのトライアスロンのメンバー","黒"],
    "シンジ":["Shinji",chara_path+"shinji_anime.jpg","赤いトライアスロンウェアに身を包んだ、30代中肉中背の男性。髪は短く、黒色の髪。ニヤリと笑っている","黒"],
    "ノリコ":["Noriko",chara_path+"noriko_anime.jpg","ブルーのランニングウェアに身を包んだ、ギャルっぽい女の子。髪はショートで青色の髪。スポーティーで笑顔が可愛い","青"],
    "ゴトウ":["Goto",  chara_path+"goto_anime.jpg","黒いウェアにサングラスをかけている。40代の男性で鬼軍曹のような厳しい雰囲気。髪はベリーショートで、真っ黒","黒"],
    "ハラダ":["Harada",chara_path+"harada_anime.jpg","オレンジのトライアスロンウェアを着た、40代の若手経営者。髪はお洒落なパーマで金髪。キラキラして自信ありげな笑みを浮かべる","金"],
    "マツイ":["Matsui",chara_path+"matsui_anime.jpg","白いランニングウェアを着たマッチョな30代男性。寡黙だが子煩悩なパパでもある。髪の毛はミディアムで緑色","紫"],
    "ケン":  ["Ken",   chara_path+"ken_anime.jpg","黒のTシャツに蛍光グリーンのマウンテンジャケットを着たバックパッカー。髪はサラサラで赤色","赤"], # トライアスロンウェアを着た痩せぎすな30代男性。自信なさげだが内なる闘志を秘めている。髪はサラサラで赤色","赤"],
    "ナオト":["Naoto", chara_path+"naoto_anime.jpg","ナオトは世界中を旅している20代の大学生。背は低いが、足が速く、引き締まった体をしている","青"],
    #"ユラ":  ["Yura",  chara_path+"yura_anime.jpg","ユラは芯が強く賢い女子高生だが、優しくいつも笑顔でみんなを和ませている。","ピンク"],
    #"旅仲間":["Travelers",chara_path+"ken_naoto.jpg","ケンとナオトの旅仲間","黒"],
    #"New":  ["new",   chara_path+"new_anime.jpg","それぞれメンバーの仲間たち","黒"],
#    "ケニー":  ["Kenny", chara_path+"kenny_anime.jpg","ケニーは世界中を旅している20代の大学生。引き締まった体をしている","赤"],
}
"""
panel_sizes = {"Small": "Small size",
        "Medium": "Medium size",
        "Big": "Big size",
        "Very big": "Very big size"}
panel_shots = {"Full": "Full shot",
        "Wide": "Wide shot",
        "Knee": "Knee shot",
        "Waist": "WWaist shot",
        "Up": "Up shot",
        "Close-up": "Close-up shot"}

import argparse
parser = argparse.ArgumentParser()

def ret_data(nums):
    image_list = []
    if os.path.isdir(results_path):
        files= os.listdir(results_path)
        for file in sorted(files):
            #file_open = open(results_path+file, 'r')
            #file_load = json.load(file_open)
            #hist_date = parse(json_load['timestamp']).strftime('%Y/%m/%d %H:%M:%S')
            if file[-4:] == ".jpg":
                image_list.append(results_path+file)
                #prompt_file = file.replace(file[-4:], ".txt")
                #print(prompt_file)
                #image_list[file] = prompt_file
    else:
        image_list = "No image files"
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

async def add_chara(LLM,llm_key, chara_up,chara_name,chara_desc,chara_color):
    print("== Chara Generation ==\n Starting {chara_name} image generation!\n")

    chara_up.save(chara_path)
    source_image = open(chara_path+chara_name+".jpg", "rb")
    image_base64 = encode_image(chara_up) # open(img_up, "rb")
    anime_prompt= f"""ここに写っている人物を、日本の90年代アニメ風の画像に変換して下さい。
        セル画のような色使いと質感で、太い輪郭線、光沢のある髪のアニメスタイルです。画像のみを出力して下さい。
        顔や表情は写真に正確に忠実に表現して下さい。
        この人物のイメージは、{chara_desc}で、髪の毛は{chara_color}色で印象的な髪型に変更して下さい。服装も全体的に{chara_color}色風にして下さい。
        ロゴや文字などは架空のものに変更して下さい。
        背景は海沿いの青空にして下さい。"""
    
    charafile = genai_image(LLM,llm_key, anime_prompt, page_size,image_quality, chara_name, source_image)
    return charafile

def plot_generate(LLM,llm_key, chara_name,page_plot, page_background, 
        page_size,image_quality, generate_page, cover_pages=1): #,img_up, dream_choice=""):
    cover_page = str(cover_pages)[-1]
    print(f"== Prompt Generation ==\n Starting {cover_pages} creation by {LLM} for {chara_name}!\n")
    print("plot_gen > "+llm_key)
    print(chara_name)

    charas_prompt= ""
    #if title_plot in ["title","cover"]:
    #if dream_choice != "":
        #system_content = f"このシステムは、テキストが提供された時に、「My Dream: {dream_choice}」をタイトルとして、主人公を{chara_name}とした物語のあらすじを300字程度で、起承転結を付けて生成します。"
        #use_plot = genai_text(LLM,llm_key, system_content, dream_choice)
    if len(page_plot) < 100:
        system_content = f"このシステムは、テキストが提供された時に、「{page_plot}」をタイトルとした物語のあらすじを300字程度で、起承転結を付けて生成します。"
        use_plot = genai_text(LLM,llm_key, system_content, page_plot)
    else:
        use_plot = page_plot
        for chara in chara_name:
            charas_prompt += f"""- 登場人物「{chara}」: {chara}の顔、表情、年齢、性別は、この画像 `<file:{all_charas[chara][1]}>` を正確に反映して下さい。
            この人物の特徴は、{all_charas[chara][2]}で、{all_charas[chara][3]}色の髪型をしています。
            """
    print("背景: "+page_background)
    print("プロット: "+use_plot)

    common_prompt = f"""# Prerequisites:
Artist Requirements: {page_styles[default_style]}
Required Background Knowledge: Ability to read and interpret character designs and storyboards
Purpose and Goal: Complete a full-color manga based on the provided text-only storyboard

# execution instruction:
Based on [#text-only storyboard] please output a full color cartoon. Please give your best effort.

# text-only storyboard:
- Character Information
    {charas_prompt}
- Overall setting:
    - Canvas size: {page_size}
    - Aspect Ratio: aspect_ratio={page_sizes[page_size]}
    - quality: {image_quality}
    - Image Size: image_size={image_qualities[image_quality]}
    - Art style: {page_styles[default_style]} (used consistently in every panel)
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

#プロット全体を通した[# テキストネームタイトル] をつけて下さい。
    #plot_prompt = f"""「{chara_name}」はこの話の主人公で、この画像のような人物です。他の登場人物としては、{charas.keys()}がいます。
    plot_prompt = f"""このストーリーの登場人物の特徴は以下のようなものです。各登場人物の服装と表情は、そのシーンに合ったものにして下さい。
{charas_prompt}
各登場人物は、そのシーンに合った服装、表情をしています。
以下の[# プロット]を元に、n=1からn={generate_page}までの、{generate_page}ページ分のプロンプトを生成して下さい。
1ページには、m=1からm={default_panel}までの{default_panel}つのPanelを作り、各Panel毎に以下のフォーマットに従って、1ページに{default_panel}つのpanelを記述したプロンプトを作成してください。
各ページの一番上に[ページタイトル]をつけて下さい。
プロットに記述したランドマークや商品、人物はなるべく正確に写実的に表現して絵柄に入れて下さい。
以下のフォーマットを元に、{generate_page}ページ分で、1ページに{default_panel}つのPanelをフォーマットに従って作って下さい。

== Prompt Format ==
■ このストーリー全体の説明: {default_desc} ■
■ ■ テキストネーム ■ ■
■ ■ [# テキストネームタイトル: ] ■ ■
# このパートの説明: {page_background}
# プロット: {use_plot}
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

    generated_prompt = ""
    cover_name = "著者名として、「Made by AniMaker」を題名の下に表示。"

    if cover_page == "0":
        direction = "\n ## Please generate one page cover with one big image exactly following the prompt with your best effort. \n"
        generated_prompt = direction + f""" Please generate Manga Cover page.
            漫画のカバーを{page_styles[default_style]}のように作って下さい。
            {page_plot}をタイトルとして、画像内の上側に大きく配置。
            {cover_name}
            そこに写っている人物の顔の特徴や表情、装飾品、髪型などは、写真を忠実に再現して下さい。
            その中心人物の人数と配置は正確に描写して下さい。
            写真を大きく中心に、印象的に配置。
            全体感は以下のプロットのテイストを入れて描写して下さい。
            ## Plot Start ##
            {use_plot}
            ## Plot End ##
            主人公はプロットに基づいた衣装、髪型、装飾にして下さい。特にその服装、格好は、プロットの内容の特徴を正確に反映して、描写して下さい。
            漫画のカバーページの背景はプロットに基づいた4コマ漫画にして下さい。そのストーリーはプロットの起承転結をつけて、背景として描いて下さい。
            {page_plot}になった将来の姿を描いて下さい。

            Background setting:
            - Image quality: crisp and clear (used consistently in every panel)
            - Font: Noto Sans JP (used consistently in every panel)
            - Panel Margins: Each panel should have a uniform margin of 10px on all four sides internally (between the artwork and the panel border).
            - Panel Border: All panels should have a consistent border, for example, a 2px solid black line.
            - Gutter Width: The space (gutter) between panels should be uniform, for example, 20px horizontally and vertically.
            - Page Margins: The entire page (canvas) should have a uniform margin of 30px around the area where panels are laid out.
            """
        # by {llms[LLM][:6].upper()}
    elif cover_page in ("5", "6"): #Figure or #3views
        direction = "\n ## Please generate the page exactly following the prompt with your best effort. \n"
        common_prompt = direction + f""" 以下の指示に基づいたページを作成して下さい.
            キャラクターを{page_styles[default_style]}のように作って下さい。
            全体感は以下のプロットのテイストを入れて描写して下さい。
            ## Plot Start ##
            {use_plot}
            ## Plot End ##
            主人公はプロットに基づいた衣装、髪型、装飾にして下さい。特にその服装、格好は、プロットの内容の特徴を正確に反映して、描写して下さい。
            プロットのストーリーになった将来の姿を描いて下さい。
            """
        if cover_page == "5": #Figure
            generated_prompt = f"""Create a 1/7 scale commercialized figurine of whole standing body of the characters in the picture, 
                in a Japanese anime style, in a real environment. The figurine is placed on a computer desk. 
                The figurine has a round transparent acrylic base, with no text on the base. 
                The content on the computer screen is a 3D modeling process of this figurine. 
                Next to the computer screen is a toy packaging box, designed in a style reminiscent of high-quality collectible figures, printed with original artwork. 
                The packaging features two-dimensional flat illustrations."""
        elif cover_page == "6": #3views
            generated_prompt = f"""この画像からフィギュアを作るために、以下の条件で画像を生成して下さい：
                - 三面図として前面・側面・背面の3つの画像を作成 
                - ランナー{runner_flag}
                - 足元に透明円形アクリル台座"""
            if title3d_flag == "YES":
                generated_prompt +=  f"頭の上に円弧状に浮かぶ立体文字「{page_plot}」と表示（アニメタイトル風）"
    else:
        system_content  = "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"
        direction = f"""\n ## Please generate {generate_page} pages image exactly following from [# Page {cover_page}] to [# Page {generate_page}] instruction,
            and put [# ページタイトル] on top of the page with your best effort.
            If there is a picture attached please use it for image generation. \n"""
        generated_prompt = direction + genai_text(LLM,llm_key, system_content, plot_prompt)

    anime_prompt = common_prompt + generated_prompt
    print(anime_prompt)
    
    return use_plot, anime_prompt 

async def plot_image_generate(LLM,llm_key, img_up,page_plot, chara_name, #page_background, 
        page_size,image_quality, generate_page, cover_pages): #, dream_choice=""):
    cover_page = str(cover_pages)[-1]
    print(f"== Prompt Image Generation ==\n {cover_pages} image by {LLM}!\n")
    print("plot_image_gen > "+llm_key)
    print(img_up)
    page_background = "このストーリーの背景はプロットに従って下さい。"
    #chara_name = default_chara
    #if img_up in charas[chara][1]:
        #chara_name = default_chara
    #else:
        #chara_name = "主人公"

    #system_content = f"このシステムは、テキストが提供された時に、「{page_plot}」をタイトルとした物語のあらすじを300字程度で、起承転結を付けて生成します。"
    #use_plot = genai_text(LLM,llm_key, system_content, page_plot)

    use_plot,prompt_out = plot_generate(LLM,llm_key, chara_name,page_plot, page_background, 
                            page_size,image_quality, generate_page, cover_pages) #img_up, dream_choice)

    resize_width = 512
    resize_proportion = resize_width / img_up.width
    img_resized = img_up.resize((int(img_up.width * resize_proportion), int(img_up.height * resize_proportion)))
    img_resized.save(img_up_path)

    print(f"== Image Generation ==\n Starting Anime image generation by {LLM}!\n")

    source_image = open(img_up_path, "rb")
    imagefile,promptfile = genai_image(LLM,llm_key, prompt_out, page_size,image_quality, chara_name, source_image)

    """
    sns_link= f"<a href="https://x.com/intent/post?text={story_name}%20
        https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F
        &url={gradio_path}{imagefile}
        &hashtags={story_name},100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>
        "
    prompt_link = f'<a href="{gradio_path}{promptfile}">Prompt</a>'
    output_link = sns_link + ' | ' + prompt_link
    """

    return use_plot, imagefile, promptfile #, output_link,  

def camera_nodetect(image):
    image.save(img_up_path)
    return "please put image chara info: "

def encode_image(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    base64_image = f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    return base64_image

def camera_detect(image,LLM):
    #apikey = os.getenv(LLM+"_KEY")
    camera_prompt = "提供された画像の中に写っている人物の、おおよその年齢、性別を類推して下さい。髪型、表情、服装を詳細に簡潔に説明して下さい。"
    image.save(img_up_path)
    image_base64 = encode_image(image)

    if LLM == "GOOGLE_API":
        gemini.configure(api_key=llm_key)
        gemini_client = gemini.GenerativeModel(llms[LLM]) #,generation_config=gemini_config)
        response = gemini_client.genai_content([image_base64, camera_prompt])
        result = response.text
        print(result)

    elif LLM == "OPENAI_API":
        openai_client = OpenAI(llm_key)
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

def raise_exception(err_msg):
    if err_msg:
        raise Exception()
    return

def flip(im):
    return np.fliplr(im)
def chara_picture(chara_name):
    return all_charas[chara_name][1]
def style_change(page_style):
    return page_styles[page_style]
def llm_change(LLM):
    return llms[LLM]
def llm_image_change(LLM):
    return llms_image[LLM]
def story_change(story_set):
    return stories[story_set][0]+stories[story_set][2]+stories[story_set][7]

def genai_text(LLM,llm_key, system_content, in_prompt):
    #llm_key = os.getenv(LLM+"_KEY") if llm_key == "" else llm_key
    #llm_model= llms[LLM]
    print("genai_text > "+llm_key)

    if LLM == "GOOGLE_API":
        gemini.configure(api_key=llm_key)
        gemini_client = gemini.GenerativeModel(llms[LLM])
            #GenerationConfig(temperature=genai_config["temperature"], max_output_tokens=genai_config["max_output_tokens"]))
            #generation_config=gemini_config)
        response= gemini_client.generate_content(in_prompt) #[image_base64, plot_prompt])
        result  = response.text
        
    elif LLM == "OPENAI_API":
        gpt_client = OpenAI(api_key=llm_key)
        #system_content = "このシステムは、画像が提供された時にそれを判別し、テキストと共に、それに合ったプロンプトを生成します。"
        response = gpt_client.chat.completions.create(
            model=llms[LLM],
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
            #generation_config = genai_config
            temperature= genai_config["temperature"],
            max_tokens = genai_config["max_output_tokens"]
        )
        result = response.choices[0].message.content
    #print("Generated Plot: "+result)
    return result
    
def genai_image(LLM,llm_key, in_prompt, page_size,image_quality, chara_name, source_image):
    #llm_key = os.getenv(LLM+"_KEY") if llm_key == "" else llm_key
    print(f"== Image Generation for {chara_name} by {llms_image[LLM]} ==\n")
    print("genai_image > "+llm_key)
    print(source_image)
    if source_image:
        source_image = open(img_up_path, "rb")
        #image_base64 = encode_image(source_image) # open(img_up, "rb")
        #img = Image.open(filename)
        with open(img_up_path, 'rb') as f:
            data = f.read()
        img = Image.open(BytesIO(data))
        image_base64 = encode_image(img)
    
    chara_images = []
    for chara in chara_name:
        if chara in all_charas.keys():
            chara_image = all_charas[chara][1]
        else:
            chara_image = img_up_path
        chara_images.append(chara_image)
    print(chara_images)
    chara_images_open = []
    for chara_image in chara_images:
        chara_images_open.append(Image.open(chara_image))
    #Image.open(chara_images[0]) 

    filename  = "anime_"+f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    imagefile = f"{results_path}{filename}_image.jpg" #{imgnum:03d}.jpg"
    promptfile= f"{results_path}{filename}_prompt.txt" #{imgnum:03d}.txt"
    raspifile = ""

    if LLM == "GOOGLE_API":
        #gemini.configure(api_key=llm_key)
        client   = genai.Client(api_key=llm_key)
        response = client.models.generate_content(
            model=llms_image[LLM], #"gemini-3-pro-image-preview"
            contents=(in_prompt, chara_images_open), #image_base64],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT','IMAGE'],
                #response_modalities=[Modality.TEXT, Modality.IMAGE],
                #temperature=1.0 #genai_config["temperature"],
                #tools=[{"google_search": {}}],
                #image_config=types.ImageConfig(
                    #aspect_ratio=page_sizes[page_size], #"16:9"
                    #image_size=image_qualities[image_quality]), #"2K"
            ),
        )
        """
        response = client.models.generate_content(model="gemini-3-pro-image-preview", #"gemini-2.0-flash-preview-image-generation",
            contents=[in_prompt, image_base64],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image'],
                #temperature=0.9 #genai_config["temperature"],
            )
        )
        response = client.chats.create(
        #response = chat.send_message(in_prompt,
            model="gemini-3-pro-image-preview",
            contents=[in_prompt, image_base64],
            #Image.open('person1.png'),
            config=types.GenerateContentConfig(
                response_modalities=['Text','Image'] #[types.Modality.TEXT, types.Modality.IMAGE],
            )
        )
        tools=[{"google_search": {}}],
        image_config=types.ImageConfig(
            aspect_ratio="9:16",
            image_size="1K"
        ),
        for part in response.candidates[0].content.parts:
            if part.text:
                with open(promptfile, "a", encoding="utf-8") as f:
                    f.write(part.text + "\n\n")
                print(part.text)
            elif part.inline_data:
                with open(promptfile, 'a', encoding='utf-8') as f:
                    f.write(in_prompt)
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save(imagefile)
                image.save("image.jpg")                
            print("ImageFile saved: " + imagefile)
        """

        for part in response.candidates[0].content.parts:
            if part.text:
                with open(promptfile, "a", encoding="utf-8") as f:
                    f.write(part.text + "\n\n")
                print(part.text)
            elif part.inline_data:
                with open(promptfile, 'a', encoding='utf-8') as f:
                    f.write(in_prompt)
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save(imagefile)
                image.save("image.jpg")                
            print("ImageFile saved: " + imagefile)

    elif LLM == "OPENAI_API":
        gpt_client = OpenAI(api_key=llm_key)
        generate_model = llms_image[LLM] #"gpt-image-1"
        response = gpt_client.images.edit(
            model  = generate_model,
            image  = chara_images_open, #source_image,
            prompt = in_prompt,
            quality= image_quality,
            size   = page_size,
            #n = generate_page
        )
        image_response = response.data[0].b64_json
        with open(promptfile, 'a', encoding='utf-8') as f:
            f.write(in_prompt)
        with open(imagefile, "wb") as f:
            f.write(base64.b64decode(image_response))
        with open("image.jpg", "wb") as f:
            f.write(base64.b64decode(image_response))
        print("ImageFile saved: "+imagefile)
        #./gradio_api/file=./results/anime_
    return imagefile, promptfile

with gr.Blocks() as animaker:
    with gr.Row():
        with gr.Accordion(label="AniMaker!", open=False):
            LLM = gr.Dropdown(choices=llms,label="0. LLM", interactive=True, value=DEF_LLM)
            llm_model = gr.Textbox(label="0. LLM Model", value=llms[DEF_LLM], interactive=True)
            LLM.change(llm_change, LLM, llm_model)
            llm_image_model = gr.Textbox(label="0. LLM Image Model", value=llms_image[DEF_LLM], interactive=True)
            LLM.change(llm_image_change, LLM, llm_image_model)
            llm_key = gr.Textbox(label="0. LLM API Key", interactive=True, value=default_key, placeholder="Paste your LLM API key here", type="password")
            page_size = gr.Dropdown(choices=page_sizes,label="Canvas size", interactive=True)
            image_quality= gr.Dropdown(choices=image_qualities,label="Image quality", interactive=True)
            generate_page= gr.Textbox(label="Generate page/s",value=default_page, interactive=True)
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
                - https://console.cloud.google.com/billing
                """)
    with gr.Row():
        with gr.Column():
            with gr.Tab("簡単アニメ作成"):
                new_up = gr.Image(label="1. Upload Photo", sources="upload",
                    type="pil", mirror_webcam=False, width=250,height=250, value=all_charas[default_name][1])
                chara_name  = gr.Dropdown(choices=all_charas, label="1. Main chara", value=default_name,  multiselect=True, allow_custom_value=True, interactive=True, scale=1)
                #dream_choice= gr.Dropdown(choices=dream_list, label="My Dream: ", interactive=True)
                #page_background = gr.Textbox(label="2. Background", value="ストーリーの背景、パート", placeholder="Enter the story background if any", interactive=True, scale=2)
                page_title     = gr.Textbox(label="2. Title/Story", placeholder="Enter the story title for anime", interactive=True, scale=2)
                cover_pages    = gr.Dropdown(choices=cover_page_list, label="3. Generate Cover (0) or Pages (1 - 4) for Anime", interactive=True)
                #cover_style= gr.Dropdown(choices=page_styles,label="4. Cover Image Style", value=default_style, interactive=True)
        
                new_btn      = gr.Button("4. AniMaker!")
                """ with gr.Accordion(open=False, label="Generated prompt"):
                    prompt_out = gr.Textbox(label="3. Prompt Out", max_lines=500,
                        placeholder ="Upload photo & plot, then edit results", interactive=True, scale=2)
                """
                output_image = gr.Image(label="4. AniMaker Image")
                output_prompt= gr.Markdown()
                    #f"""<a href="https://x.com/intent/post?text={page_title}%20
                    #https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F
                    #&url={gradio_path}{output_image}
                    #&hashtags={page_title},100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>
                    # | <a href="{gradio_path}{output_prompt}">Prompt</a>""")
                with gr.Accordion(open=False, label="Generated results"):
                    sys_msg   = gr.Markdown(label="System message") 
                    use_plot  = gr.Markdown(label="Generated plot")
                    prompt_out= gr.Markdown(label="Generated prompt")
                new_btn.click(fn=plot_image_generate, inputs=[LLM,llm_key, new_up,page_title, chara_name, #page_background, 
                        page_size,image_quality, generate_page, cover_pages], #, dream_choice], 
                    outputs=[use_plot,output_image,output_prompt], api_name="plot_image_generate")

            with gr.Tab("あらすじから作成"):
                story_set = gr.Dropdown(choices=stories, label="0. Story set", interactive=True, scale=1, value=default_story)
                story_desc= gr.Markdown(default_desc)
                story_set.change(story_change, story_set, story_desc)
                #img_up = gr.Image(label="1. Chara Photo", sources="upload",
                    #type="pil", mirror_webcam=False, width=250,height=250, value=charas[default_chara][1])     
                chara_name = gr.Dropdown(choices=all_charas, label="1. Chara", interactive=True, scale=1, multiselect=True, value=default_name) #, #Textbox(label="Chara Name", interactive=True)
                #chara_name.change(chara_picture, chara_name, img_up)
                #dream_choice= gr.Dropdown(choices=dream_list, label="My Dream: ", interactive=True)
                page_background = gr.Textbox(label="2. Background", value="ストーリー中のパート、ステップ", placeholder="Enter the story background if any", interactive=True, scale=2)
                use_plot   = gr.Textbox(label="2. Plot", placeholder="Put your story plot here", interactive=True, scale=2)
                cover_pages= gr.Dropdown(choices=cover_page_list, label="3. Generate Cover (0) or Pages (1 - 4) for Anime", interactive=True)
                plot_btn   = gr.Button("3. Generate Prompt")
                prompt_out = gr.Textbox(label="3. Prompt Out", max_lines=500,
                    placeholder ="Upload photo & plot, then edit results", interactive=True, scale=2)
                plot_btn.click(fn=plot_generate, inputs=[LLM,llm_key, chara_name,use_plot, page_background, #img_up,
                            page_size,image_quality, generate_page, cover_pages], # dream_choice], 
                            outputs=[use_plot, prompt_out], api_name="plot_generate")

                promptfile_genimage   = gr.Markdown()
                output_prompt_genimage= gr.Markdown()
                        #f"""<a href="https://x.com/intent/post?text={page_title}%20
                        #https%3A%2F%2Fktrips.net%2F100-days-to-ironman%2F
                        #&url={gradio_path}{output_image}
                        #&hashtags={page_title},100Days2Ironman,ironman&openExternalBrowser=1">Post SNS</a>
                        # | <a href="{gradio_path}{output_prompt}">Prompt</a>""")

                genimage_btn = gr.Button("4. Generate Image!")
                output_genimage = gr.Image(label="4. Generated Image")
                genimage_btn.click(fn=genai_image, inputs=[LLM,llm_key, prompt_out, page_size,image_quality, chara_name], #, cover_pages], 
                    outputs=[output_genimage,promptfile_genimage], api_name="genai_image")

                with gr.Accordion(open=False, label="Generated results"):
                    sys_msg   = gr.Markdown(label="System message") 
                    use_plot  = gr.Markdown(label="Generated plot")
                    prompt_out= gr.Markdown(label="Generated prompt")
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
                style_desc= gr.Markdown(value=page_styles[default_style])
                page_style.change(style_change, page_style, style_desc)
                #page_story= gr.Dropdown(choices=page_storys,label="Generate/Manual", value=default_story, interactive=True)
                #generate_page= gr.Textbox(label="Generate page/s",value=default_page, interactive=True)
                #page_size = gr.Dropdown(choices=page_sizes,label="Canvas size",  value=default_size, interactive=True)
                #image_quality= gr.Dropdown(choices=image_qualities,label="Image quality", value=default_quality, interactive=True)
                page_panel  = gr.Textbox(label="# of panels in a page",value=default_panel, interactive=True)
                num_steps= gr.Slider(minimum=1,maximum=20,value=default_steps,step=1, label="Steps",interactive=True)
            with gr.Accordion(label="Charactors:", open=False):
                for chara in all_charas:
                    with gr.Row(chara):
                        chara_image= gr.Image(all_charas[chara][1], label=chara)
                        chara_chara= gr.Markdown(all_charas[chara][2])
                with gr.Row():
                    chara_name = gr.Textbox(label="Add Chara Name", interactive=True)
                    chara_desc = gr.Textbox(label="Chara Description", placeholder="Add photo and chara description", interactive=True)
                    chara_color= gr.Dropdown(choices=colors, label="Chara Color", value=default_color, interactive=True)
                    chara_up = gr.Image(label="Person's photo", sources="upload",
                        type="pil", mirror_webcam=False, width=200,height=200)
                    chara_btn = gr.Button("Add Anime Chara")
                    chara_btn.click(fn=add_chara, inputs=[LLM,llm_key, chara_up,chara_name,chara_desc,chara_color],
                        outputs=[gr.Image(label="Generated Anime Chara",width=200,height=200)], api_name="add_chara")
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
            gr.Gallery(ret_data(5), columns=5, show_label=True, show_download_button=True, show_share_button=True, allow_preview=True)

parser = argparse.ArgumentParser(description="Gradio UI for Anime Maker")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
args = parser.parse_args()
animaker.launch(server_name=args.ip,server_port=args.port) #, auth=(animaker_usr,animaker_pswd))
