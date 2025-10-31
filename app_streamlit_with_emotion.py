# app_streamlit_with_emotion_extended.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from typing import List

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🌎 NASA Climate Engagement Dashboard", layout="wide")
st.title("🌍 NASA Climate Engagement & Sentiment Dashboard — Advanced Edition")
st.write("Now includes sentiment, emotion, trend, engagement, and topic analysis of climate-related conversations.")

# ------------------ SETTINGS ------------------
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

@st.cache_resource
def load_emotion_pipeline(model_name: str):
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
    if pipe is None:
        return [None] * len(texts)
    try:
        raw = pipe(texts, truncation=True)
        out = []
        for item in raw:
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
    with st.spinner("Loading emotion model (first run may take a while)..."):
        emo_pipe = load_emotion_pipeline(EMOTION_MODEL_NAME)
else:
    emo_pipe = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year

    df = compute_sentiment(df)

    # Emotion detection
    if enable_emotion and emo_pipe is not None:
        st.info("Running emotion detection...")
        texts = df["text"].astype(str).tolist()
        emotion_scores = predict_emotion_batch(emo_pipe, texts)
        if any(emotion_scores):
            labels = sorted(set(lab for m in emotion_scores if m for lab in m.keys()))
            for lab in labels:
                df[f"emo_{lab}"] = [m.get(lab, 0.0) if m else 0.0 for m in emotion_scores]
            df["top_emotion"] = [max(m, key=m.get) if m else None for m in emotion_scores]
        else:
            df["top_emotion"] = None
    else:
        df["top_emotion"] = None

    # ------------------ TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sentiment & Emotion", 
        "Trend Analysis", 
        "Engagement Analysis", 
        "Topic Modeling",
        "Predictive Modeling"
    ])

    # ---------- TAB 1: SENTIMENT & EMOTION ----------
    with tab1:
        st.subheader("📈 Sentiment & Emotion Overview")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="sentiment_label", palette="coolwarm", order=["Positive","Neutral","Negative"], ax=ax)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
        with col2:
            if df["top_emotion"].notna().any():
                fig2, ax2 = plt.subplots()
                sns.countplot(data=df, x="top_emotion", palette="mako", order=df["top_emotion"].value_counts().index, ax=ax2)
                plt.xticks(rotation=45)
                ax2.set_title("Top Emotions")
                st.pyplot(fig2)

        st.subheader("☁️ Word Clouds by Sentiment")
        wc1, wc2, wc3 = st.columns(3)
        for label, col in zip(["Positive","Neutral","Negative"], [wc1, wc2, wc3]):
            with col:
                wc = generate_wordcloud(df[df["sentiment_label"]==label]["text"])
                if wc is not None: st.image(wc.to_array(), caption=label)
            # ---------- TOP POSTS BY SENTIMENT / EMOTION ----------
        st.subheader("🏆 Top Posts by Sentiment / Emotion")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top Positive Posts (by likes)")
            top_pos = df[df["sentiment_label"] == "Positive"].sort_values("likesCount", ascending=False).head(5)
            st.dataframe(top_pos[["date", "likesCount", "commentsCount", "text", "polarity", "top_emotion"]])

        with col2:
            st.markdown("### Top Negative Posts (by likes)")
            top_neg = df[df["sentiment_label"] == "Negative"].sort_values("likesCount", ascending=False).head(5)
            st.dataframe(top_neg[["date", "likesCount", "commentsCount", "text", "polarity", "top_emotion"]])

    # ---------- TOP POSTS BY EMOTION ----------
        if df["top_emotion"].notna().any():
            st.subheader("🔍 Top posts per detected emotion")
            for emo in df["top_emotion"].dropna().unique():
                st.markdown(f"**Top posts for emotion: {emo}**")
                emo_df = df[df["top_emotion"] == emo].sort_values("likesCount", ascending=False).head(3)
                st.dataframe(emo_df[["date", "likesCount", "commentsCount", "text"]])
    
        
    # ---------- TAB 2: TREND ANALYSIS ----------
    with tab2:
        st.subheader("📊 Sentiment Trend Over Time")
        trend = df.groupby(pd.Grouper(key='date', freq='M'))['polarity'].mean().reset_index()
        st.line_chart(trend, x='date', y='polarity', height=300)
        st.write("Average monthly sentiment trend shows how climate conversation tone changes over time.")

    # ---------- TAB 3: ENGAGEMENT ANALYSIS ----------
    with tab3:
        st.subheader("💬 Engagement Correlations")
        corr = df[['likesCount','commentsCount','polarity']].corr()
        st.write(corr)
        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

        st.subheader("📊 Likes vs Comments (Colored by Sentiment)")
        fig = px.scatter(df, x='likesCount', y='commentsCount', color='sentiment_label',
                         title="Engagement Scatter Plot", hover_data=['text'])
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 4: TOPIC MODELING ----------
    with tab4:
        st.subheader("🧠 Topic Modeling (LDA on Comments)")
        vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
        X = vectorizer.fit_transform(df['text'].dropna())
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)
        words = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            st.markdown(f"**Topic {idx+1}:** " + ", ".join([words[i] for i in topic.argsort()[-10:]]))

    # ---------- TAB 5: PREDICTIVE MODEL ----------
    with tab5:
        st.subheader("🤖 Predict Likes from Post Text")
        X = df[["text","commentsCount"]]
        y = df["likesCount"].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("reg", LinearRegression())
        ])
        X_train_text = X_train["text"].astype(str) + " comments:" + X_train["commentsCount"].astype(str)
        X_test_text = X_test["text"].astype(str) + " comments:" + X_test["commentsCount"].astype(str)
        pipeline.fit(X_train_text, y_train)
        preds = pipeline.predict(X_test_text)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        st.success(f"Model trained — R²: {r2:.3f}, MAE: {mae:.3f}")

        user_text = st.text_area("Enter new post text:", height=100)
        user_comments = st.number_input("Number of comments:", min_value=0, value=0)
        if st.button("Predict Likes"):
            input_text = [user_text + " comments:" + str(user_comments)]
            pred = pipeline.predict(input_text)[0]
            st.markdown(f"**Predicted likes:** {pred:.1f}")

    # ------------------ DOWNLOAD ------------------
    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download annotated dataset", csv_bytes, "climate_nasa_extended.csv", mime="text/csv")

else:
    st.info("👆 Upload `climate_nasa.csv` to start the full analysis.")
