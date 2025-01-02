# Introduction

Here is a simple Telegram bot to recommend arXiv papers daily, obtain your preference ratings and update the recommender models given the preference ratings. 

# Install
## Get Bot token
Create a telegram bot following the instruction [here](https://core.telegram.org/bots/tutorial). Then you get a bot token and store it in `TELEGRAM_BOT_TOKEN` as an environment variable.

## Get Chat ID
Then you can get your chat id. First randomly chat with the bot you just created in telegram, then query https://api.telegram.org/bot{your bot token}/getUpdates, and look for chat ids. Save the chat id into `TELEGRAM_BOT_CHAT_ID` as another environment variable. 

## Install dependency  
```
pip install -r requirement.txt
```

## Run the bot
```
python arxiv_checker.py --check
```

## Update the model. 
```
python preference_model.py
```
It will save the model to `pytorch_preference_model.pt` and `tfidf_vectorizer.joblib` (as TF-IDF vectorizer). Then you restart `arxiv_checker.py` to load the updated models. 



