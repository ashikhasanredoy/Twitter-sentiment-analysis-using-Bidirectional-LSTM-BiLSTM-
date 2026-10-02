import re


def clean_tweet(text):
    """
    Clean tweet text by:
    - converting to lowercase
    - removing URLs
    - removing mentions
    - removing hashtags symbol
    - removing punctuation
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove @mentions and # symbol
    text = re.sub(r"@\w+|#", "", text)

    # Remove punctuation/special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
