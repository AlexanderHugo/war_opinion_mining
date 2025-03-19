import praw
import json
from datetime import datetime
from publisher import RabbitmqPublisher
from dataStructures import PostDataStructure


class RedditAPI:
    __client = 'zICxCNUGj9zcE1vjsoMvkw'
    __secret = 'zuIwG6Nh1zpQxz_56hMxdMygL6-CkQ'
    __user_agent = 'WarOpMining'


    def __init__(self, category="all"):
        self.__api = praw.Reddit(
            client_id = self.__client,
            client_secret = self.__secret,
            user_agent = self.__user_agent
        )

        self.publisher = RabbitmqPublisher()

        self.topic = self.__api.subreddit(category)

    def search(self, keyword_list, limit, sort="hot"):
        results = self.topic.search(
            ' AND '.join(f'"{x}"' for x in keyword_list),
            limit = limit,
            sort = sort
        )

        # Convert the ListingGenerator results to JSON format
        posts = []
        for submission in results:
            submission.comments.replace_more(limit=None)
            post_data: PostDataStructure = {
                "id": submission.id,
                "title": submission.title,
                "upvote_ratio": submission.upvote_ratio,
                "author": str(submission.author),  # Author can be None, so convert to string
                "created_utc": datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "score": submission.score,
                "url": submission.url,
                "selftext": submission.selftext,
                "num_comments": submission.num_comments,
                "comments": [comment.body for comment in submission.comments.list()]
            }
            self.publisher.send_message(post_data)
            posts.append(post_data)

        # Return JSON format data
        return posts