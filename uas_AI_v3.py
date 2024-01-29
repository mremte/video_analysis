import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
from wordcloud import WordCloud
import mudghol as sm
from PIL import Image

def video_to_text(video_path):
    # Extract audio from the video file
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("audio.wav", codec='pcm_s16le')
    audio_clip.close()

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Read the audio from the WAV file
    with sr.AudioFile("audio.wav") as source:
        print("Recognizing text...")
        audio = recognizer.record(source)

        try:
            # Recognize speech in Indonesian
            text_indonesian = recognizer.recognize_google(audio, language='id-ID')
            return text_indonesian  # Return the recognized text

        except sr.UnknownValueError:
            print("Sorry, could not recognize the speech.")
            return None  # Return None or handle the unknown value error as needed

        except sr.RequestError as e:
            print(f"Error with the API request: {e}")
            return None  # Return None or handle the request error as needed

def generate_wordcloud(text):
    # Generate a Word Cloud from the recognized text
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Convert the Word Cloud to a PIL Image
    wordcloud_image = wordcloud.to_image()

    # Display the Word Cloud using Streamlit
    st.subheader("Word Cloud")
    st.image(wordcloud_image, use_column_width=True)

def main():
    st.title("Video and Sentiment Analysis")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded file
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the uploaded video
        hasil = video_to_text(video_path)

        if hasil is not None:
            st.success("Text successfully recognized:")
            st.write(hasil)

            # Perform Sentiment Analysis
            hasil_sentimen = sm.sentimen(hasil)
            st.success("Sentiment Analysis:")
            st.write(hasil_sentimen)

            # Generate and display Word Cloud
            generate_wordcloud(hasil)

if __name__ == "__main__":
    main()
