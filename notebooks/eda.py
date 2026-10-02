import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})

FIGURES_DIR = "results/figures"
DATA_PATH = "data/raw/training.csv"


def run_eda(data_path=DATA_PATH, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding="latin1")

    print("\n--- DATA OVERVIEW ---")
    print("Head:\n", df.head())
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:\n", df.isna().sum())
    print("\nSentiment value counts:\n", df["sentiment"].value_counts())

    # 1. Sentiment Distribution
    print("Generating Sentiment Distribution plot...")
    plt.figure(figsize=(8, 5))
    palette = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
    ax = sns.countplot(
        x="sentiment",
        data=df,
        palette=palette,
        order=["negative", "neutral", "positive"]
    )
    plt.title("Distribution of Tweet Sentiments", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Tweet Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=10,
            color="#2c3e50",
            xytext=(0, 4),
            textcoords="offset points"
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"), dpi=300)
    plt.close()

    # 2. Sentiment Trends by Time of Day
    if "Time of Tweet" in df.columns:
        print("Generating Sentiment Trends by Time plot...")
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x="Time of Tweet",
            hue="sentiment",
            data=df,
            palette=palette,
            hue_order=["negative", "neutral", "positive"]
        )
        plt.title("Sentiment Trends by Time of Day", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Time of Day", fontsize=12)
        plt.ylabel("Number of Tweets", fontsize=12)
        plt.legend(title="Sentiment", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_time.png"), dpi=300)
        plt.close()

    # 3. Sentiment by Age
    if "Age of User" in df.columns:
        print("Generating Age Distribution vs Sentiment plot...")
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=df,
            x="Age of User",
            hue="sentiment",
            multiple="stack",
            palette=palette,
            hue_order=["negative", "neutral", "positive"],
            shrink=0.8
        )
        plt.title("Age Distribution vs Sentiment", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Age Group", fontsize=12)
        plt.ylabel("Tweet Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_age.png"), dpi=300)
        plt.close()

    # 4. Positive Wordcloud
    print("Generating Positive Wordcloud...")
    pos_text = " ".join(t for t in df[df["sentiment"] == "positive"]["text"].dropna().astype(str))
    pos_wc = WordCloud(
        background_color="white",
        width=1000,
        height=500,
        colormap="Greens"
    ).generate(pos_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(pos_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Common Words in Positive Tweets", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "positive_wordcloud.png"), dpi=300)
    plt.close()

    # 5. Negative Wordcloud
    print("Generating Negative Wordcloud...")
    neg_text = " ".join(t for t in df[df["sentiment"] == "negative"]["text"].dropna().astype(str))
    neg_wc = WordCloud(
        background_color="black",
        width=1000,
        height=500,
        colormap="Reds"
    ).generate(neg_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(neg_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Common Words in Negative Tweets", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "negative_wordcloud.png"), dpi=300)
    plt.close()

    # 6. Top 10 Countries
    if "Country" in df.columns:
        print("Generating Top 10 Countries plot...")
        plt.figure(figsize=(9, 8))
        top_countries = df["Country"].value_counts().head(10)
        colors = sns.color_palette("Set2", 10)
        top_countries.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        plt.title("Top 10 Countries in Dataset", fontsize=14, pad=12, fontweight="bold")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "country_distribution.png"), dpi=300)
        plt.close()

    print(f"\nAll EDA figures successfully created in '{output_dir}'.")


if __name__ == "__main__":
    run_eda()
