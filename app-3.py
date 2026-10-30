import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, sep="\t", names=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, target_names=["ham", "spam"], output_dict=True
        ),
        "cm": confusion_matrix(y_test, y_pred),
    }
    return model, vectorizer, metrics


st.title("📧 Spam Message Classifier")
st.write("Detects whether a message is spam using TF-IDF and Naive Bayes.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

model, vectorizer, metrics = train_model(df)

tab_predict, tab_perf, tab_data = st.tabs(["Predict", "Model performance", "Dataset"])

with tab_predict:
    message = st.text_area("Enter a message:", height=120,
                           placeholder="e.g. Congratulations! You won a free iPhone...")
    if st.button("Classify"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            vec = vectorizer.transform([message])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            if pred == 1:
                st.error(f"🚨 SPAM (confidence: {prob[1]:.1%})")
            else:
                st.success(f"✅ NOT SPAM (confidence: {prob[0]:.1%})")

with tab_perf:
    st.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    report_df = pd.DataFrame(metrics["report"]).T.round(3)
    st.dataframe(report_df)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(metrics["cm"], display_labels=["ham", "spam"]).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

with tab_data:
    st.write(f"Total messages: {len(df)}")
    st.bar_chart(df["label"].value_counts())
    st.dataframe(df[["label", "text"]].head(50))
