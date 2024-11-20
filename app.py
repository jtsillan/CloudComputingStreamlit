import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@st.cache_data
def download_vader():
    return nltk.download('vader_lexicon')


def sentiment_scores(text):
    # create sentiment Intensity Analyzer object
    analyzer = SentimentIntensityAnalyzer()

    # provides dictionary with pos, neg, neu & compound key and their values
    scores = analyzer.polarity_scores(text)
    positive = scores['pos'] * 100
    negative = scores['neg'] * 100
    neutral = scores['neu'] * 100
    compound = scores['compound'] * 100
    return scores, positive, neutral, negative, compound


def clear_fields():
    st.session_state["input_text"] = ""


def main():
    st.title("Sentiment Analysis App")
    # Load an image from a file path
    image_path = "emojis.png"
    st.image(image_path, use_container_width=True)
    # Create a form with text area box for the user to input the string
    with st.form("submit_form"):
        user_input = st.text_area(label="Text here", key="input_text", height=150)
        col1, col2 = st.columns(2)            
        with col1:
            analyze_button = st.form_submit_button("Analyze") 
        with col2:
            st.form_submit_button("Clear", on_click=clear_fields)
        
        if analyze_button:
            _, c2, _ = st.columns(3)
            scores, positive, neutral, negative, compound = sentiment_scores(user_input)
            st.write(scores)
            with c2:
                if compound >= 5:
                    st.image("emoji_happy.png", caption=f"Analyze was positive {round(positive, 2)}%")
                elif compound <= -5:
                    st.image("emoji_sad.png", caption=f"Analyze was negative {round(negative, 2)}%")
                else:
                    st.image("emoji_neutral.png", caption=f"Analyze was neutral {round(neutral, 2)}%")
                         

if __name__=="__main__":
    download_vader()
    main()