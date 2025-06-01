# BOOKA
## App for Book and Prices with AI Agent
- AI agent browse and comapre prices for books, movies, music, and all other BOOKA!

## Sample app on the web:
BOOKA.Ktrips.net - 
<img src="https://github.com/user-attachments/assets/85bc7877-f968-43e3-abb9-c2947eb5b486" width="300">

## Deploy to gcloud run

### Clone booka github

```
$ git clone https://github.com/ktrips/booka.git
$ cd booka
$ gcloud services enable run.googleapis.com
$ mkdir results
$ vi .env
```
### Pepare AI API keys in .env file
Get API keys for OpenAI, Google, and/or Anthropic.
Create .env file with the API keys:
```text:.env
chatgpt_key=XXXX
gemini_key=YYYY
claude_key=ZZZZ
```

### Go and get Google project ID and enable Google APIs

### Create Docker environment
```
$ docker build --platform=linux/amd64 -t asia-northeast1-docker.pkg.dev/[your_project]/booka:v1 .
$ docker run -p 8080:8080 -e CHROME_PERSISTENT_SESSION=true asia-northeast1-docker.pkg.dev/[your_project]/booka:v1
```

### Push and deploy docker environment to Gloud

```
$ docker push asia-northeast1-docker.pkg.dev/[your_project]/booka:v1
$ gcloud run deploy booka --image=asia-northeast1-docker.pkg.dev/[your_project]/booka:v1 --memory=1Gi --port=8080 --allow-unauthenticated --platform=managed --region=asia-northeast1
```
