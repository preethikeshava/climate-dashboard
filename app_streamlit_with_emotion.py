# app_streamlit_with_emotion.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🌎 NASA Climate Engagement Dashboard", layout="wide")
st.title("🌍 NASA Climate Engagement & Sentiment Dashboard — with Emotion Detection")
st.write("Text sentiment + emotion detection, word clouds, top posts, and likes-prediction model.")

# ------------------ QUICK SETTINGS ------------------
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"  # compact emotion model
# Alternative (bigger) models exist; pick a smaller one for CPU if necessary.

@st.cache_resource
def load_emotion_pipeline(model_name: str):
    """
    Load and cache the HF emotion classification pipeline.
    Might take time and require internet first-run.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        emo_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)
        return emo_pipe
    except Exception as e:
        st.warning(f"Unable to load emotion model: {e}")
        return None

# ------------------ HELPERS ------------------
def compute_sentiment(df: pd.DataFrame):
    df["polarity"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["sentiment_label"] = df["polarity"].apply(
        lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.1 else "Neutral")
    )
    return df

def generate_wordcloud(text_series: pd.Series, bg_color="white"):
    text = " ".join([str(t) for t in text_series.tolist() if pd.notna(t)])
    if not text.strip():
        return None
    wc = WordCloud(width=400, height=250, background_color=bg_color, colormap="Set2").generate(text)
    return wc

def predict_emotion_batch(pipe, texts: List[str], top_k:int=3):
    """
    Returns list of dicts: [{label:score, ...}, ...] for each input text.
    pipe returns list of lists of dicts (label, score)
    We convert to mapping label->score
    """
    if pipe is None:
        return [None] * len(texts)
    try:
        raw = pipe(texts, truncation=True)
        out = []
        for item in raw:
            # item is list of dicts for each label
            mapped = {d['label']: float(d['score']) for d in item}
            out.append(mapped)
        return out
    except Exception as e:
        st.warning(f"Emotion inference failed: {e}")
        return [None] * len(texts)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload your `climate_nasa.csv`", type=["csv"])
enable_emotion = st.sidebar.checkbox("Enable emotion detection (HuggingFace)", value=False)
if enable_emotion:
    with st.spinner("Loading emotion model (may take a minute first run)..."):
        emo_pipe = load_emotion_pipeline(EMOTION_MODEL_NAME)
        if emo_pipe is None:
            st.sidebar.error("Emotion model not available — will run without emotion detection.")
else:
    emo_pipe = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Basic cleaning
    df = df.dropna(subset=["text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year

    # Sentiment
    df = compute_sentiment(df)

    # Emotion detection (optional) — batch predict to speed up
    if enable_emotion and emo_pipe is not None:
        st.subheader("🔎 Emotion detection (this may take a moment)...")
        texts = df["text"].astype(str).tolist()
        emotion_scores = predict_emotion_batch(emo_pipe, texts)
        # convert mapping to separate columns for main emotions found (take union)
        if any(emotion_scores):
            # get label set
            labels = set()
            for m in emotion_scores:
                if m:
                    labels.update(m.keys())
            labels = sorted(labels)
            for lab in labels:
                df[f"emo_{lab}"] = [ (m.get(lab, 0.0) if m else 0.0) for m in emotion_scores ]
            # set top emotion label column
            def top_emotion(m):
                if not m:
                    return None
                best = max(m.items(), key=lambda x: x[1])
                return best[0]
            df["top_emotion"] = [ (top_emotion(m) if m else None) for m in emotion_scores ]
            st.success("Emotion detection complete.")
        else:
            st.info("No emotion outputs. Running without emotions.")
            df["top_emotion"] = None
    else:
        df["top_emotion"] = None

    # --- Visuals: Sentiment and Emotion distributions
    st.subheader("📈 Sentiment & Emotion Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="sentiment_label", palette="coolwarm", order=["Positive","Neutral","Negative"], ax=ax)
        ax.set_title("Sentiment Breakdown")
        st.pyplot(fig)
    with col2:
        if df["top_emotion"].notna().any():
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x="top_emotion", palette="mako", order=df["top_emotion"].value_counts().index, ax=ax2)
            ax2.set_title("Top Emotion Breakdown")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        else:
            st.info("Top emotion column not available — enable emotion detection to see distributions.")

    # Word clouds by sentiment
    st.subheader("☁️ Word Clouds by Sentiment")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown("**Positive posts**")
        wc = generate_wordcloud(df[df["sentiment_label"]=="Positive"]["text"])
        if wc is not None: st.image(wc.to_array())
    with wc2:
        st.markdown("**Neutral posts**")
        wc = generate_wordcloud(df[df["sentiment_label"]=="Neutral"]["text"])
        if wc is not None: st.image(wc.to_array())
    with wc3:
        st.markdown("**Negative posts**")
        wc = generate_wordcloud(df[df["sentiment_label"]=="Negative"]["text"])
        if wc is not None: st.image(wc.to_array())

    # Top posts by sentiment and emotion
    st.subheader("🏆 Top Posts by Sentiment / Emotion")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Top Positive Posts (by likes)")
        top_pos = df[df["sentiment_label"]=="Positive"].sort_values("likesCount", ascending=False).head(5)
        st.write(top_pos[["date","likesCount","commentsCount","text","polarity","top_emotion"]])
    with c2:
        st.markdown("### Top Negative Posts (by likes)")
        top_neg = df[df["sentiment_label"]=="Negative"].sort_values("likesCount", ascending=False).head(5)
        st.write(top_neg[["date","likesCount","commentsCount","text","polarity","top_emotion"]])

    # If emotion available, show top posts per top emotion
    if df["top_emotion"].notna().any():
        st.subheader("🔎 Top posts per detected emotion")
        emotions = df["top_emotion"].dropna().unique().tolist()
        for emo in emotions:
            st.markdown(f"**Top posts for emotion: {emo}**")
            sel = df[df["top_emotion"]==emo].sort_values("likesCount", ascending=False).head(3)
            st.write(sel[["date","likesCount","commentsCount","text"]])

    # ------------------ ML MODEL: PREDICT LIKES ------------------
    st.subheader("🤖 Train Predictive Model (Likes from Text + Comments)")
    X = df[["text","commentsCount"]]
    y = df["likesCount"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("reg", LinearRegression())
    ])
    X_train_text = X_train["text"].astype(str) + " comments:" + X_train["commentsCount"].astype(str)
    X_test_text = X_test["text"].astype(str) + " comments:" + X_test["commentsCount"].astype(str)

    with st.spinner("Training likes-prediction model..."):
        pipeline.fit(X_train_text, y_train)
        preds = pipeline.predict(X_test_text)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
    st.success(f"Model trained — R²: {r2:.3f}, MAE: {mae:.3f}")

    # ------------------ New post prediction + emotion ------------------
    st.subheader("🔮 Predict Likes & Emotions for a New Post")
    user_text = st.text_area("Enter post text (new):", height=120)
    user_comments = st.number_input("Number of comments:", min_value=0, step=1, value=0)
    if st.button("Predict & Analyze"):
        if not user_text.strip():
            st.error("Please provide some post text.")
        else:
            # Likes prediction
            user_input = [user_text + " comments:" + str(user_comments)]
            pred_likes = pipeline.predict(user_input)[0]
            # sentiment
            polarity = TextBlob(user_text).sentiment.polarity
            sentiment_label = "Positive" if polarity>0.1 else ("Negative" if polarity<-0.1 else "Neutral")
            st.markdown(f"**Predicted likes:** {pred_likes:.1f}")
            st.markdown(f"**Sentiment:** {sentiment_label} (polarity={polarity:.2f})")

            # emotion (if available)
            if enable_emotion and emo_pipe is not None:
                with st.spinner("Running emotion classifier..."):
                    emo_scores = predict_emotion_batch(emo_pipe, [user_text])[0]
                    if emo_scores:
                        # show top 3 emotions
                        sorted_em = sorted(emo_scores.items(), key=lambda x: x[1], reverse=True)
                        st.markdown("**Emotion scores (top 3):**")
                        for k,v in sorted_em[:3]:
                            st.write(f"- {k}: {v:.2f}")
                    else:
                        st.info("Emotion model couldn't provide results.")
            else:
                st.info("Emotion detection disabled (toggle in sidebar).")

    # ------------------ Export / small summary ------------------
    st.markdown("---")
    st.write("Download an annotated CSV (sentiment/emotion columns added):")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download annotated CSV", csv_bytes, file_name="climate_nasa_annotated.csv", mime="text/csv")

else:
    st.info("👆 Upload the `climate_nasa.csv` file to start analysis.")
