"""
Example script showing how to use the Whisper transcription functionality
"""

from main import transcribe_audio
import sys


def example_usage():
    print("Whisper Transcription Example")
    print("=" * 30)
    
    # Example usage (you would need to provide an actual audio file)
    audio_file = "example_audio.wav"  # Replace with your audio file
    
    print(f"Attempting to transcribe: {audio_file}")
    print("Note: This is just an example. You need to provide an actual audio file.")
    
    try:
        # Basic transcription
        result = transcribe_audio(audio_file, model_size="base")
        
        # Extract text from result
        text = result["text"]
        print("Transcription result (text only):")
        print(text)
        
        print("\nSegments with timestamps:")
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text_segment = segment["text"]
            print(f"[{start:.2f}s -> {end:.2f}s] {text_segment}")
            
        print("\n" + "="*30)
        print("Advanced options example:")
        
        # Transcription with advanced options
        result_advanced = transcribe_audio(
            audio_file, 
            model_size="base",
            temperature=0.0,
            word_timestamps=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        print("Advanced transcription completed with word-level timestamps option.")
        
        print("\n" + "="*30)
        print("For batch processing multiple files or directories, use the command-line interface:")
        print("python main.py /path/to/directory --output-dir /path/to/output --parallel")
        
        print("\n" + "="*30)
        print("For speaker diarization (who spoke when), you need a Hugging Face token:")
        print("python main.py audio.wav --diarize --hf-token YOUR_TOKEN --output-format json")
        
    except FileNotFoundError:
        print(f"File {audio_file} not found. Please provide a valid audio file.")
        print("\nTo test this script, you can:")
        print("1. Download an audio file and update the audio_file variable")
        print("2. Run: python main.py <audio_file>")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    example_usage()