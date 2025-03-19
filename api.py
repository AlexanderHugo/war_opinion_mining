from reddit import RedditAPI

api = RedditAPI()

responses = api.search(
        keyword_list = ['war', 'israel', 'palestine'],
        limit = 500
    )