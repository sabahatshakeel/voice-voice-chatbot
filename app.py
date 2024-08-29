import gradio as gr
import os
from groq import Groq
import whisper
from gtts import gTTS
import tempfile
import traceback

# Initialize the Groq client with your API key
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize Whisper model
whisper_model = whisper.load_model("base")

def chatbot(audio_input):
    try:
        # Step 1: Transcribe the audio input using Whisper
        transcription = whisper_model.transcribe(audio_input)["text"]
        print(f"User said: {transcription}")

        # Step 2: Get a response from Llama 8B using Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": transcription,
                }
            ],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Bot response: {response_text}")

        # Step 3: Convert the response text to speech using GTTS
        tts = gTTS(text=response_text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)

        # Return the transcription, response text, and audio file path
        return transcription, response_text, temp_file.name

    except Exception as e:
        # Print the error traceback for debugging purposes
        print(f"Error: {traceback.format_exc()}")
        # Return error messages to the Gradio interface
        return " ", " ", None

# Define the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(type="filepath", label="Speak to the bot"),  # Removed source parameter
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Response"), gr.Audio(label="Response Audio")],
    live=True  # Simulates real-time interaction by processing each input as it is received
)

# Launch the Gradio interface
iface.launch()