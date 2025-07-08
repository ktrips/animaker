from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import config # MessagingAPIのアクセストークンなど
from google.cloud import datastore
import datetime, os
from gradio_client import Client, handle_file

app = Flask(__name__)
client = datastore.Client()
line_bot_api = LineBotApi(config.token)
handler = WebhookHandler(config.secret)
#from dotenv import load_dotenv
#load_dotenv()
LLM    = "GOOGLE_API"
llm_key= config.GOOGLE_API_KEY #os.getenv(LLM+"_KEY")
animaker_usr = config.ANIMAKER_USR #os.getenv("ANIMAKER_USR")
animaker_pswd= config.ANIMAKER_PSWD #os.getenv("ANIMAKER_PSWD")

img_up     = handle_file('./shinji.jpg')
page_plot  = "Hello!!"
cover_pages= "Cover 0"
title_plot = "Cover 0"
charaname  = "Jp 90s"
chara_name = "シンジ"
api_url    = "https://anime.ktrips.net/"

def plot_image_generate():
    gradio_client = Client(api_url) #, auth=(animaker_usr,animaker_pswd))
    result1 = gradio_client.predict(
		LLM, # = llm_default,  # "OPENAI_API", "GOOGLE_API", "ANTHOLOPIC"
		llm_key, #=llm_key_str,    
		img_up, #=img_file,
		page_plot,
		cover_pages, #="Cover 0",
		api_name="/plot_image_generate"
    )
    #print(result1)
    return result1

def plot_generate(page_plot):
    gradio_client = Client(api_url) #, auth=(animaker_usr,animaker_pswd))
    result2 = gradio_client.predict(
        LLM, #=llm_default,
        llm_key, #=llm_key_str,
        img_up, #=img_file,
        chara_name,
        page_plot, #="Hello!!",
        cover_pages, #="Cover 0",
        api_name="/plot_generate"
    )
    #print(result2)
    return result2

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_key = client.key("TestTable", user_id) # kindとidを引数にKeyを取得
    user_entity = client.get(user_key) # Keyを引数にEntityを取得
    message_id  = event.message.id
    message_type= event.message.type

    if user_entity is None:
        user_entity = datastore.Entity(key=user_key, exclude_from_indexes=("timestamp",))
        msg = "はじめまして！写真をアップロードするか、描きたいアニメの仮題を入れて下さい！"
    else:
        msg = message_type # + message_text
        """
        if message_type != "text":
            msg += "テキスト以外を受け取りました！"
            #message_content = line_bot_api.get_message_content(message_id)
            #orgFile = message_content.getBlob()
        else:
        #if message_type == "text": 
            message_text= event.message.text
            #gentext  = plot_generate(message_text)
            timestamp= user_entity["timestamp"]
            ts = datetime.datetime.fromtimestamp(timestamp/1000)
            msg +="{}年{}月{}日{}時{}分以来ですね！".format(ts.year, ts.month, ts.day, ts.hour, ts.minute)
            #msg= msg + message_text #"\n" + gentext
        """
    user_entity.update({ # Entityの更新
        "timestamp": event.timestamp
    })
    client.put(user_entity) # 引数のEntityをDatastoreに保存
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=msg))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8180)))
