import argparse
import os
import tempfile
import json
import whisper
from pathlib import Path
import glob
from multiprocessing import Pool, cpu_count
import functools
import torch


def validate_audio_file(audio_path):
    """Validate if the audio file exists and is accessible"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not os.path.isfile(audio_path):
        raise ValueError(f"Path is not a file: {audio_path}")
    
    # Check if file is readable
    if not os.access(audio_path, os.R_OK):
        raise PermissionError(f"File is not readable: {audio_path}")
    
    # Check file size (not too large)
    file_size = os.path.getsize(audio_path)
    # 1 GB limit as an example, Whisper can handle longer files but processing time increases
    if file_size > 1024 * 1024 * 1024:  # 1GB
        raise ValueError(f"Audio file is too large (>1GB): {audio_path}")


def validate_output_path(output_path):
    """Validate if the output path is valid and writable"""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")


def get_audio_files(audio_path):
    """
    Get a list of audio files from a path (file or directory).
    
    Args:
        audio_path (str): Path to an audio file or directory
    
    Returns:
        list: List of audio file paths
    """
    path = Path(audio_path)
    
    if path.is_file():
        # Single file
        return [str(path)]
    elif path.is_dir():
        # Directory - find all audio files
        audio_extensions = ['*.mp3', '*.mp4', '*.m4a', '*.wav', '*.m4p', '*.mpga', '*.webm', '*.mkv', '*.avi', '*.mov', '*.flv', '*.wmv', '*.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(str(path / ext), recursive=True))
            audio_files.extend(glob.glob(str(path / ext.upper()), recursive=True))
        return sorted(audio_files)
    else:
        raise ValueError(f"Path does not exist: {audio_path}")


def transcribe_single_audio(args):
    """
    Transcribe a single audio file. This function is designed to work with multiprocessing.
    
    Args:
        args (tuple): Tuple containing (audio_path, model_size, language, translate_to_english, **options)
    
    Returns:
        dict: Result dictionary or None if error occurs
    """
    (audio_path, model_size, language, translate_to_english, output_format, output_dir, 
     temperature, compression_ratio_threshold, logprob_threshold, no_speech_threshold,
     condition_on_previous_text, initial_prompt, word_timestamps, clip_timestamps,
     prepend_punctuations, append_punctuations, diarize, huggingface_token) = args
    
    try:
        print(f"Processing: {audio_path}")
        
        # Validate the audio file
        validate_audio_file(audio_path)
        
        # Load the Whisper model
        print(f"Loading Whisper model ({model_size}) for {audio_path}")
        model = whisper.load_model(model_size)
        
        # Prepare transcription options for Whisper model (excluding diarization params)
        whisper_options = {
            "language": language,
            "task": "translate" if translate_to_english else "transcribe",
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
            "clip_timestamps": clip_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations
        }
        
        # Call transcribe_audio with specific diarization parameters
        result = transcribe_audio(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            translate_to_english=translate_to_english,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            clip_timestamps=clip_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            diarize=diarize,
            huggingface_token=huggingface_token
        )
        
        if translate_to_english:
            print(f"Translation to English completed for {audio_path}")
        else:
            print(f"Transcription completed for {audio_path}")
        
        # Get the base filename without extension
        base_name = Path(audio_path).stem
        
        # Determine output path
        if output_dir:
            output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
        else:
            # Use same directory as input file
            input_dir = os.path.dirname(audio_path)
            output_path = os.path.join(input_dir, f"{base_name}.{output_format}")
        
        # Validate output path
        validate_output_path(output_path)
        
        # Handle output based on format
        if output_format == "txt":
            output_content = result["text"]
        elif output_format in ["srt", "vtt", "tsv"]:
            # Create a temporary directory for the writer
            with tempfile.TemporaryDirectory() as temp_dir:
                from whisper.utils import get_writer
                writer = get_writer(output_format, temp_dir)
                
                # Use a temporary filename in the temp directory
                temp_audio_path = os.path.join(temp_dir, "temp_audio")
                
                # Write the output using the writer
                writer(result, temp_audio_path)
                
                # Read the content from the generated file
                temp_output_path = f"{temp_audio_path}.{output_format}"
                with open(temp_output_path, 'r', encoding='utf-8') as temp_file:
                    output_content = temp_file.read()
        elif output_format == "json":
            output_content = json.dumps(result, indent=2, ensure_ascii=False)
        
        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"Transcription saved to: {output_path}")
        
        return {"status": "success", "file": audio_path, "output": output_path}
    
    except Exception as e:
        error_msg = f"Error processing {audio_path}: {str(e)}"
        print(error_msg)
        return {"status": "error", "file": audio_path, "error": str(e)}


def process_audio_batch_parallel(audio_paths, model_size="base", language=None, translate_to_english=False, 
                                 output_format="txt", output_dir=None, num_workers=None, diarize=False, 
                                 huggingface_token=None, **kwargs):
    """
    Process a batch of audio files in parallel.
    
    Args:
        audio_paths (list): List of audio file paths to transcribe
        model_size (str): Size of the Whisper model to use (default: "base")
        language (str): Language of the audio (optional, Whisper will auto-detect if not provided)
        translate_to_english (bool): Whether to translate the result to English (default: False)
        output_format (str): Output format (default: "txt")
        output_dir (str): Directory to save output files (default: same as input file)
        num_workers (int): Number of worker processes (default: number of CPU cores)
        diarize (bool): Whether to perform speaker diarization (default: False)
        huggingface_token (str): Hugging Face access token for the diarization model
        **kwargs: Additional options to pass to transcribe_single_audio
    """
    if num_workers is None:
        # Use number of CPU cores minus 1 to leave one core free
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Processing {len(audio_paths)} files using {num_workers} worker processes")
    
    # Prepare arguments for each audio file
    args_list = []
    for audio_path in audio_paths:
        args = (audio_path, model_size, language, translate_to_english, output_format, 
                output_dir, kwargs.get('temperature', 0.0),
                kwargs.get('compression_ratio_threshold', 2.4),
                kwargs.get('logprob_threshold', -1.0),
                kwargs.get('no_speech_threshold', 0.6),
                kwargs.get('condition_on_previous_text', True),
                kwargs.get('initial_prompt', None),
                kwargs.get('word_timestamps', False),
                kwargs.get('clip_timestamps', '0'),
                kwargs.get('prepend_punctuations', "\"'\"¿([{-"),
                kwargs.get('append_punctuations', "\"'.。,，!！?？:：\")]}、'"),
                diarize,
                huggingface_token)
        args_list.append(args)
    
    # Process files in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(transcribe_single_audio, args_list)
    
    # Report results
    successful = [r for r in results if r and r["status"] == "success"]
    failed = [r for r in results if r and r["status"] == "error"]
    
    print(f"\nProcessing completed: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f['file']}: {f['error']}")


def convert_audio_for_diarization(audio_path):
    """
    Convert audio file to a format compatible with diarization (16kHz, mono).
    
    Args:
        audio_path (str): Path to the original audio file
    
    Returns:
        str: Path to the converted audio file (or original file if no conversion needed)
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path
    
    try:
        # Check the audio properties using ffprobe
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a:0', audio_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Could not probe audio file {audio_path}, using original file")
            return audio_path
        
        # Parse the ffprobe output to get sample rate and channels
        lines = result.stdout.split('\n')
        sample_rate = None
        channels = None
        
        for line in lines:
            if line.startswith('sample_rate='):
                sample_rate = int(line.split('=')[1])
            elif line.startswith('channels='):
                channels = int(line.split('=')[1])
        
        # Check if conversion is needed
        needs_conversion = False
        if sample_rate is None or sample_rate != 16000:
            needs_conversion = True
        if channels is None or channels != 1:
            needs_conversion = True
        
        if not needs_conversion:
            # File is already in compatible format
            return audio_path
        
        # Create temporary file for converted audio with .wav extension
        temp_dir = tempfile.gettempdir()
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        temp_filename = os.path.join(temp_dir, f"temp_diarization_{basename}.wav")
        
        # Convert audio using ffmpeg
        convert_result = subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',     # Set to mono
            '-c:a', 'pcm_s16le',  # Use PCM S16LE codec (most compatible)
            temp_filename
        ], capture_output=True, text=True)
        
        if convert_result.returncode == 0:
            print(f"Audio converted for diarization: {audio_path} -> {temp_filename}")
            return temp_filename
        else:
            print(f"Failed to convert audio for diarization, using original: {convert_result.stderr}")
            return audio_path
            
    except Exception as e:
        print(f"Error converting audio for diarization: {str(e)}, using original file")
        return audio_path


def diarize_audio(audio_path, huggingface_token=None):
    """
    Perform speaker diarization on the audio file to identify who spoke when.
    
    Args:
        audio_path (str): Path to the audio file to diarize
        huggingface_token (str): Hugging Face access token for the diarization model
    
    Returns:
        pyannote.core.Annotation: Diarization result with speaker turn information
    """
    try:
        import torch
        from pyannote.audio import Pipeline
        
        # Convert audio to compatible format if needed
        converted_audio_path = convert_audio_for_diarization(audio_path)
        
        # Load the diarization pipeline
        if huggingface_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=huggingface_token
            )
        else:
            raise ValueError("Hugging Face token is required for diarization")
        
        # Send pipeline to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print(f"Using GPU for diarization: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU for diarization")
        
        # Apply the pipeline to the audio file
        print(f"Diarizing audio: {converted_audio_path}")
        diarization = pipeline(converted_audio_path)
        
        print("Diarization completed.")
        
        # Clean up temporary file if it was created
        if converted_audio_path != audio_path:
            import os
            try:
                os.remove(converted_audio_path)
                print(f"Temporary converted audio file cleaned up: {converted_audio_path}")
            except:
                pass  # Ignore errors when cleaning up temporary file
        
        return diarization
    except ImportError:
        print("Warning: pyannote.audio not installed. Diarization will be skipped.")
        print("To enable diarization, install it with: pip install pyannote.audio")
        return None
    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        return None


def combine_transcription_diarization(transcription_result, diarization_result):
    """
    Combine transcription and diarization results to assign speakers to transcription segments.
    
    Args:
        transcription_result (dict): Result from Whisper transcription
        diarization_result: Result from diarization pipeline
        
    Returns:
        list: Segments with both text and speaker information
    """
    if diarization_result is None:
        # Return original segments if no diarization was performed
        return transcription_result["segments"]
    
    # Create a mapping between time ranges and speakers
    # Each segment from Whisper will be matched with the corresponding speaker from diarization
    combined_segments = []
    
    for segment in transcription_result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Find which speaker was talking during this time range
        speakers_in_segment = []
        for speech_turn, _, speaker in diarization_result.speaker_diarization.itertracks(yield_label=True):
            # Check if there's an overlap between the transcription segment and diarization turn
            if speech_turn.start <= end_time and speech_turn.end >= start_time:
                overlap_start = max(speech_turn.start, start_time)
                overlap_end = min(speech_turn.end, end_time)
                overlap_duration = overlap_end - overlap_start
                
                speakers_in_segment.append({
                    "speaker": speaker,
                    "start": overlap_start,
                    "end": overlap_end,
                    "duration": overlap_duration
                })
        
        # Add speaker information to the segment
        segment_with_speaker = segment.copy()
        segment_with_speaker["speakers"] = speakers_in_segment
        combined_segments.append(segment_with_speaker)
    
    return combined_segments


def transcribe_audio(audio_path, model_size="base", language=None, translate_to_english=False, temperature=0.0, 
                     compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6,
                     condition_on_previous_text=True, initial_prompt=None, word_timestamps=False, 
                     clip_timestamps="0", prepend_punctuations="\"'\"¿([{-", 
                     append_punctuations="\"'.。,，!！?？:：\")]}、'", diarize=False, huggingface_token=None):
    """
    Transcribes audio to text using OpenAI's Whisper ASR model.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model_size (str): Size of the Whisper model to use (default: "base")
        language (str): Language of the audio (optional, Whisper will auto-detect if not provided)
        translate_to_english (bool): Whether to translate the result to English (default: False)
        temperature (float): Temperature for sampling (default: 0.0)
        compression_ratio_threshold (float): Threshold for compression ratio (default: 2.4)
        logprob_threshold (float): Threshold for log probability (default: -1.0)
        no_speech_threshold (float): Threshold for no speech probability (default: 0.6)
        condition_on_previous_text (bool): Whether to condition on previous text (default: True)
        initial_prompt (str): Optional text to provide as a prompt (default: None)
        word_timestamps (bool): Enable word-level timestamps (default: False)
        clip_timestamps (str): Specific clips to process (default: "0")
        prepend_punctuations (str): Punctuations to prepend to words (default: "\"'\"¿([{-")
        append_punctuations (str): Punctuations to append to words (default: "\"'.。,，!！?？:：\")]}、'")
        diarize (bool): Whether to perform speaker diarization (default: False)
        huggingface_token (str): Hugging Face access token for the diarization model
    
    Returns:
        dict: The result dictionary containing transcription and segments
    """
    # Validate the audio file
    validate_audio_file(audio_path)
    
    # Load the Whisper model with GPU support
    print(f"Loading Whisper model ({model_size})...")
    try:
        # Determine device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = whisper.load_model(model_size).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{model_size}': {str(e)}")
    
    print(f"Transcribing audio: {audio_path}")
    
    try:
        # Prepare transcription options
        options = {
            "language": language,
            "task": "translate" if translate_to_english else "transcribe",
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
            "clip_timestamps": clip_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations
        }
        
        # Transcribe or translate the audio
        result = model.transcribe(audio_path, **options)
        
        if translate_to_english:
            print("Translation to English completed.")
        else:
            print("Transcription completed.")
        
        # Perform diarization if requested
        if diarize:
            diarization_result = diarize_audio(audio_path, huggingface_token)
            if diarization_result is not None:
                # Combine transcription and diarization results
                result["segments"] = combine_transcription_diarization(result, diarization_result)
                print("Diarization completed and combined with transcription.")
            else:
                print("Diarization was requested but could not be performed.")
        
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")
    
    return result


def process_audio_batch(audio_paths, model_size="base", language=None, translate_to_english=False, 
                        output_format="txt", output_dir=None, parallel=False, num_workers=None, 
                        diarize=False, huggingface_token=None, **kwargs):
    """
    Process a batch of audio files.
    
    Args:
        audio_paths (list): List of audio file paths to transcribe
        model_size (str): Size of the Whisper model to use (default: "base")
        language (str): Language of the audio (optional, Whisper will auto-detect if not provided)
        translate_to_english (bool): Whether to translate the result to English (default: False)
        output_format (str): Output format (default: "txt")
        output_dir (str): Directory to save output files (default: same as input file)
        parallel (bool): Whether to process files in parallel (default: False)
        num_workers (int): Number of worker processes if parallel=True (default: number of CPU cores - 1)
        diarize (bool): Whether to perform speaker diarization (default: False)
        huggingface_token (str): Hugging Face access token for the diarization model
        **kwargs: Additional options to pass to transcribe_audio
    """
    if parallel and len(audio_paths) > 1:
        # Process in parallel
        process_audio_batch_parallel(
            audio_paths=audio_paths,
            model_size=model_size,
            language=language,
            translate_to_english=translate_to_english,
            output_format=output_format,
            output_dir=output_dir,
            num_workers=num_workers,
            diarize=diarize,
            huggingface_token=huggingface_token,
            **kwargs
        )
    else:
        # Process sequentially
        for audio_path in audio_paths:
            try:
                print(f"\nProcessing: {audio_path}")
                
                # Get the base filename without extension
                base_name = Path(audio_path).stem
                
                # Determine output path
                if output_dir:
                    output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
                else:
                    # Use same directory as input file
                    input_dir = os.path.dirname(audio_path)
                    output_path = os.path.join(input_dir, f"{base_name}.{output_format}")
                
                # Transcribe the audio
                result = transcribe_audio(
                    audio_path=audio_path,
                    model_size=model_size,
                    language=language,
                    translate_to_english=translate_to_english,
                    diarize=diarize,
                    huggingface_token=huggingface_token,
                    **kwargs
                )
                
                # Handle output based on format
                if output_format == "txt":
                    # Check if diarization was performed
                    if diarize and "segments" in result and len(result["segments"]) > 0:
                        # Check if the first segment has speaker information
                        if "speakers" in result["segments"][0]:
                            # Create a diarized transcript
                            output_content = ""
                            for i, segment in enumerate(result["segments"]):
                                start_time = segment["start"]
                                end_time = segment["end"]
                                text = segment["text"].strip()
                                
                                # Get speaker information
                                speakers = segment.get("speakers", [])
                                if speakers:
                                    speaker = speakers[0]["speaker"]
                                else:
                                    speaker = "SPEAKER_UNKNOWN"
                                
                                # Format: [start_time - end_time] SPEAKER: text
                                output_content += f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}\n"
                        else:
                            output_content = result["text"]
                    else:
                        output_content = result["text"]
                elif output_format in ["srt", "vtt", "tsv"]:
                    # Create a temporary directory for the writer
                    with tempfile.TemporaryDirectory() as temp_dir:
                        from whisper.utils import get_writer
                        writer = get_writer(output_format, temp_dir)
                        
                        # Use a temporary filename in the temp directory
                        temp_audio_path = os.path.join(temp_dir, "temp_audio")
                        
                        # Write the output using the writer
                        writer(result, temp_audio_path)
                        
                        # Read the content from the generated file
                        temp_output_path = f"{temp_audio_path}.{output_format}"
                        with open(temp_output_path, 'r', encoding='utf-8') as temp_file:
                            output_content = temp_file.read()
                elif output_format == "json":
                    output_content = json.dumps(result, indent=2, ensure_ascii=False)
                
                # Write the output file
                # Validate output path
                validate_output_path(output_path)
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_content)
                    
                    print(f"Transcription saved to: {output_path}")
                except Exception as e:
                    raise IOError(f"Failed to write output file: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text using OpenAI's Whisper model"
    )
    parser.add_argument(
        "audio_path", 
        help="Path to the audio file or directory to transcribe"
    )
    parser.add_argument(
        "--model", 
        choices=["tiny", "base", "small", "medium", "large", "large-v2"], 
        default="base",
        help="Whisper model size to use (default: base)"
    )
    parser.add_argument(
        "--language",
        help="Language of the audio (e.g., 'en', 'pt', 'es'). If not provided, Whisper will auto-detect."
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate the result to English."
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "srt", "vtt", "tsv", "json"],
        default="txt",
        help="Output format: txt (text), srt (SubRip), vtt (WebVTT), tsv (tab-separated), json (JSON format)"
    )
    parser.add_argument(
        "--output",
        help="Output file to save the transcription. If not provided, prints to console for single files. For directories, use --output-dir instead."
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files when processing a directory. If not provided, saves to the same directory as each input file."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple files in parallel using multiprocessing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes to use for parallel processing (default: number of CPU cores - 1)."
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Perform speaker diarization to identify who spoke when."
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face access token for diarization model (required if using --diarize)."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for processing if available (default: True if GPU is available)."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force use of CPU instead of GPU."
    )
    
    # Segmentation options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)"
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Threshold for compression ratio (default: 2.4)"
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Threshold for log probability (default: -1.0)"
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Threshold for no speech probability (default: 0.6)"
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=True,
        help="Condition on previous text (default: True)"
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_true",
        help="Do not condition on previous text"
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt"
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps"
    )
    parser.add_argument(
        "--clip-timestamps",
        default="0",
        help="Specific clips to process (default: \"0\")"
    )
    
    args = parser.parse_args()
    
    # Handle mutually exclusive arguments
    if args.no_condition_on_previous_text:
        condition_on_previous_text = False
    else:
        condition_on_previous_text = args.condition_on_previous_text

    try:
        # Validate arguments
        if args.language and len(args.language) != 2:
            print(f"Warning: Language code '{args.language}' does not look like a standard 2-letter code.")
        
        if args.diarize and not args.hf_token:
            print("Error: --hf-token is required when using --diarize")
            return 1
        
        # Determine GPU usage
        if args.no_gpu:
            # Force CPU usage
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
            print("Using CPU for processing (GPU disabled by user).")
        elif args.use_gpu or (torch.cuda.is_available() and not args.no_gpu):
            print(f"Using GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
        else:
            print("Using CPU for processing.")
        
        # Get list of audio files to process
        audio_files = get_audio_files(args.audio_path)
        
        if not audio_files:
            print(f"No audio files found in: {args.audio_path}")
            return 1
        
        print(f"Found {len(audio_files)} audio file(s) to process")
        
        # Check if we're processing a single file with console output (no --output specified)
        if len(audio_files) == 1 and not args.output and not args.output_dir:
            # Process single file and output to console
            result = transcribe_audio(
                audio_path=audio_files[0],
                model_size=args.model,
                language=args.language,
                translate_to_english=args.translate,
                temperature=args.temperature,
                compression_ratio_threshold=args.compression_ratio_threshold,
                logprob_threshold=args.logprob_threshold,
                no_speech_threshold=args.no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=args.initial_prompt,
                word_timestamps=args.word_timestamps,
                clip_timestamps=args.clip_timestamps,
                diarize=args.diarize,
                huggingface_token=args.hf_token
            )
            
            # Handle output based on format
            if args.output_format == "txt":
                # Check if diarization was performed
                if args.diarize and "segments" in result and len(result["segments"]) > 0:
                    # Check if the first segment has speaker information
                    if "speakers" in result["segments"][0]:
                        # Create a diarized transcript
                        output_content = ""
                        for i, segment in enumerate(result["segments"]):
                            start_time = segment["start"]
                            end_time = segment["end"]
                            text = segment["text"].strip()
                            
                            # Get speaker information
                            speakers = segment.get("speakers", [])
                            if speakers:
                                speaker = speakers[0]["speaker"]
                            else:
                                speaker = "SPEAKER_UNKNOWN"
                            
                            # Format: [start_time - end_time] SPEAKER: text
                            output_content += f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}\n"
                    else:
                        output_content = result["text"]
                else:
                    output_content = result["text"]
            elif args.output_format in ["srt", "vtt", "tsv"]:
                # Create a temporary directory for the writer
                with tempfile.TemporaryDirectory() as temp_dir:
                    from whisper.utils import get_writer
                    writer = get_writer(args.output_format, temp_dir)
                    
                    # Use a temporary filename in the temp directory
                    temp_audio_path = os.path.join(temp_dir, "temp_audio")
                    
                    # Write the output using the writer
                    writer(result, temp_audio_path)
                    
                    # Read the content from the generated file
                    temp_output_path = f"{temp_audio_path}.{args.output_format}"
                    with open(temp_output_path, 'r', encoding='utf-8') as temp_file:
                        output_content = temp_file.read()
            elif args.output_format == "json":
                output_content = json.dumps(result, indent=2, ensure_ascii=False)
            
            print(f"\nTranscription ({args.output_format}):")
            print(output_content)
        else:
            # Process one or more files with file output
            process_audio_batch(
                audio_paths=audio_files,
                model_size=args.model,
                language=args.language,
                translate_to_english=args.translate,
                output_format=args.output_format,
                output_dir=args.output_dir,
                parallel=args.parallel,
                num_workers=args.workers,
                diarize=args.diarize,
                huggingface_token=args.hf_token,
                temperature=args.temperature,
                compression_ratio_threshold=args.compression_ratio_threshold,
                logprob_threshold=args.logprob_threshold,
                no_speech_threshold=args.no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=args.initial_prompt,
                word_timestamps=args.word_timestamps,
                clip_timestamps=args.clip_timestamps
            )

    except FileNotFoundError as e:
        print(f"File error: {str(e)}")
        return 1
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return 1
    except PermissionError as e:
        print(f"Permission error: {str(e)}")
        return 1
    except RuntimeError as e:
        print(f"Runtime error: {str(e)}")
        return 1
    except IOError as e:
        print(f"I/O error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
