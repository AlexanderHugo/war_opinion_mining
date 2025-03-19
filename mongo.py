from typing import List, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from dataStructures import PostDataStructure, EnhancedPostDataStructure

class WarOpMongoDB:
    local_storage: List[EnhancedPostDataStructure] = []
    _instance: Optional['WarOpMongoDB'] = None
    client: MongoClient
    db: Database
    collection: Collection

    def __new__(cls) -> 'WarOpMongoDB':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self) -> None:
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['warOpMiningDB']
        self.collection = self.db.get_collection("reddit")

    def insert_new_post(self, post_data):
        try:
            # Create a document with the necessary fields
            new_post = {
                "id": post_data.get('id'),
                "title": post_data.get('title'),
                "selftext": post_data.get('selftext'),
                "author": post_data.get('author'),
                "created_utc": post_data.get('created_utc'),
                "num_comments": post_data.get('num_comments'),
                "score": post_data.get('score'),
                "upvote_ratio": post_data.get('upvote_ratio'),
                "url": post_data.get('url'),
                "comments": post_data.get('comments', [])
            }
            result = self.collection.insert_one(new_post)
            print(f"Inserted post with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"Error inserting new post: {e}")
            return None

    def findPostById(self, value) -> Optional[PostDataStructure]:
        post = self.collection.find_one({"id": value})
        if post is None:
            return None
        return PostDataStructure(
            id=post.get('id', ''),  # Changed from '_id' to 'id'
            title=post.get('title', ''),
            upvote_ratio=post.get('upvote_ratio', 0.0),
            author=post.get('author', None),
            created_utc=post.get('created_utc', 0),
            score=post.get('score', 0),
            url=post.get('url', ''),
            selftext=post.get('selftext', ''),
            num_comments=post.get('num_comments', 0),
            comments=post.get('comments', [])
        )

    def findAllPosts(self) -> List[PostDataStructure]:
        posts = self.collection.find()
        return [
            PostDataStructure(
                id=post.get('_id', ''),
                title=post.get('title', ''),
                upvote_ratio=post.get('upvote_ratio', 0.0),
                author=post.get('author', None),
                created_utc=post.get('created_utc', 0),
                score=post.get('score', 0),
                url=post.get('url', ''),
                selftext=post.get('selftext', ''),
                num_comments=post.get('num_comments', 0),
                comments=post.get('comments', [])
            ) for post in posts
        ]

    def findEnhancedPostById(self, postId):
        return next((x for x in self.local_storage if x.id == postId), None)

    def getAllEnhancedPost(self):
        return self.local_storage

    def saveEnhancedPost(self, post: EnhancedPostDataStructure):
        existing_post = self.findEnhancedPostById(post.id)
        if existing_post is None:
            self.local_storage.append(post)
        else:
            # Replace the existing post with the new one
            index = self.local_storage.index(existing_post)
            self.local_storage[index] = post

        print(f"Local Storage updated. Total posts: {len(self.local_storage)}")
        return None
    
    def saveBucketEnhancedPosts(self, posts: List[EnhancedPostDataStructure]):
        for post in posts:
            self.saveBucketEnhancedPosts(post)

        