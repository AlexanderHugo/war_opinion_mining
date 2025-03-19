import praw

client = 'zICxCNUGj9zcE1vjsoMvkw'
secret = 'zuIwG6Nh1zpQxz_56hMxdMygL6-CkQ'
user_agent = 'WarOpMining'

client = praw.Reddit(
    client_id = client,
    client_secret = secret,
    user_agent = user_agent
)

topic = client.subreddit("all")

results = topic.search(
    'news AND brazil',
    limit = 5,
    sort = "hot"
)

for item in results:
    print(f"Item {item.id}: {item.title}")