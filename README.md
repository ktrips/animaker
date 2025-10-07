# AniMaker
## App for Anime Making with AI Agent
- AI agent to generate anime pictures 

## Sample app on the web:
ANIME.Ktrips.net - 
<img src="https://github.com/user-attachments/assets/050603ba-9d83-441b-b7cc-f826fee340d7" width="400">

## Deploy to gcloud run

### Clone animaker github

```
$ git clone https://github.com/ktrips/animaker.git
$ cd animaker
$ gcloud services enable run.googleapis.com
$ mkdir results chara image
$ vi .env
```
### Pepare AI API keys in .env file
Get API keys for OpenAI, Google, and/or Anthropic.
Create .env file with the API keys:
```text:.env
OPENAI_API_KEY=XXXX
GOOGLE_API_KEY=YYYY
ANTHOLOPOC_API_KEY=ZZZZ

ANIMAKER_USR=uuu
ANIMAKER_PSWD=ppp
```

### Go and get Google project ID and enable Google APIs

### Create Docker environment
```
$ docker build --platform=linux/amd64 -t asia-northeast1-docker.pkg.dev/[your_project]/animaker-repo/animaker:v1 .
$ docker run -p 8080:8080 asia-northeast1-docker.pkg.dev/[your_project]/animaker-repo/animaker:v1
```

### Push and deploy docker environment to GCloud

```
$ gcloud artifacts repositories create animaker-repo --repository-format=docker --location=asia-northeast1 --description="Docker repository for Cloud Run Anime Maker"
$ gcloud auth configure-docker asia-northeast1-docker.pkg.dev
$ docker push asia-northeast1-docker.pkg.dev/[your_project]/animaker-repo/animaker:v1
$ gcloud run deploy animaker --image=asia-northeast1-docker.pkg.dev/[your_project]/animaker-repo/animaker:v1 --memory=1Gi --port=8080 --allow-unauthenticated --platform=managed --region=asia-northeast1
```
Now you can access https://animaker-xxxx.asia-northeast1.run.app or AniMaker.Ktrips.net! Enjoy!

## How to use AniMaker:
### 0. LLMとAPI Keyをセット
### New Story:
1. 写真をアップして
2. 適当なタイトル(例: シンガポールで大冒険)を入れて
3. アニメ風、アメコミ風などを選べば
4. アニメの表紙と4ページのマンガがAniMaker!
### Chara Plot:
0. キャラを事前登録
1. キャラを選んで
2. あらすじを入れると
3. 4ページ分のプロンプト生成
4. それを微修正して、5コマのマンガがAniMaker!
#### AI Usages:
- https://platform.openai.com/usage
- https://console.cloud.google.com/billing
