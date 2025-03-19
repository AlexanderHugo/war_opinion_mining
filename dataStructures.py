from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any

@dataclass
class PostDataStructure:
    id: str
    title: str
    upvote_ratio: float
    author: Optional[str]
    created_utc: str
    score: int
    url: str
    selftext: str
    num_comments: int
    comments: List[str]

    def __iter__(self):
        return iter([
            self.id,
            self.title,
            self.upvote_ratio,
            self.author,
            self.created_utc,
            self.score,
            self.url,
            self.selftext,
            self.num_comments,
            self.comments
        ])



@dataclass
class EnhancedPostDataStructure:
    id: str
    title: str
    upvote_ratio: float
    author: Optional[str]
    created_utc: str
    score: int
    url: str
    selftext: str
    num_comments: int
    comments: List[str]
    sentiment_score: dict
    overall_sentiment_score: dict
    topic: dict

    def __iter__(self):
        return iter([
            self.id,
            self.title,
            self.upvote_ratio,
            self.author,
            self.created_utc,
            self.score,
            self.url,
            self.selftext,
            self.num_comments,
            self.comments,
            self.sentiment_score,
            self.overall_sentiment_score,
            self.topic
        ])