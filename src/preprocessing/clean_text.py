import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_HASHTAG_PATTERN = re.compile(r"[@#]\w+|#")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = URL_PATTERN.sub("", text)
    text = MENTION_HASHTAG_PATTERN.sub("", text)
    text = PUNCTUATION_PATTERN.sub("", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()
