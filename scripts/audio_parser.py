#!/usr/bin/env python3
"""
Video to Text Transcription Script

This script extracts audio from video files and transcribes it to text.
Supports multiple video formats and provides options for output formatting.

Requirements:
    pip install moviepy speechrecognition pydub

Optional (for better performance):
    pip install torch whisper-openai

Usage:
    python video_transcriber.py input_video.mp4 output_transcript.txt
    python video_transcriber.py input_video.mp4  # outputs to input_video_transcript.txt
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import logging

# Core dependencies
try:
    import moviepy.editor as mp
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install moviepy speechrecognition pydub")
    sys.exit(1)

# Optional dependencies for better performance
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class VideoTranscriber:
    """Main class for video transcription with multiple engine options."""
    
    def __init__(self, engine: str = "google", model_size: str = "base"):
        """
        Initialize the transcriber.
        
        Args:
            engine: Transcription engine ("google", "whisper", "sphinx")
            model_size: Whisper model size if using whisper ("tiny", "base", "small", "medium", "large")
        """
        self.engine = engine.lower()
        self.model_size = model_size
        self.recognizer = sr.Recognizer()
        
        # Initialize Whisper model if available and selected
        if self.engine == "whisper" and WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model(model_size)
                logging.info(f"Loaded Whisper model: {model_size}")
            except Exception as e:
                logging.warning(f"Failed to load Whisper model: {e}")
                self.engine = "google"
                logging.info("Falling back to Google Speech Recognition")
        elif self.engine == "whisper" and not WHISPER_AVAILABLE:
            logging.warning("Whisper not available, falling back to Google Speech Recognition")
            self.engine = "google"
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            audio_path: Path to output audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"Extracting audio from {video_path}")
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio.close()
            video.close()
            logging.info(f"Audio extracted to {audio_path}")
            return True
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> AudioSegment:
        """
        Preprocess audio for better transcription results.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioSegment: Processed audio
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz for optimal speech recognition
            audio = audio.set_frame_rate(16000)
            
            return audio
        except Exception as e:
            logging.error(f"Error preprocessing audio: {e}")
            return AudioSegment.from_file(audio_path)
    
    def split_audio_on_silence(self, audio: AudioSegment, min_silence_len: int = 500, 
                              silence_thresh: int = -40) -> List[AudioSegment]:
        """
        Split audio into chunks based on silence.
        
        Args:
            audio: AudioSegment to split
            min_silence_len: Minimum length of silence in ms
            silence_thresh: Silence threshold in dB
            
        Returns:
            List[AudioSegment]: List of audio chunks
        """
        try:
            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=200  # Keep some silence for context
            )
            
            # If no splits found, split into fixed-size chunks
            if len(chunks) <= 1:
                chunk_length = 30000  # 30 seconds
                chunks = []
                for i in range(0, len(audio), chunk_length):
                    chunks.append(audio[i:i + chunk_length])
            
            logging.info(f"Split audio into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logging.error(f"Error splitting audio: {e}")
            return [audio]
    
    def transcribe_chunk_google(self, audio_chunk: AudioSegment) -> str:
        """Transcribe audio chunk using Google Speech Recognition."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_chunk.export(temp_file.name, format="wav")
                
                with sr.AudioFile(temp_file.name) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                
                os.unlink(temp_file.name)
                return text
        except sr.UnknownValueError:
            return "[INAUDIBLE]"
        except sr.RequestError as e:
            logging.error(f"Google API error: {e}")
            return "[ERROR]"
        except Exception as e:
            logging.error(f"Error transcribing chunk: {e}")
            return "[ERROR]"
    
    def transcribe_chunk_whisper(self, audio_chunk: AudioSegment) -> str:
        """Transcribe audio chunk using Whisper."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_chunk.export(temp_file.name, format="wav")
                result = self.whisper_model.transcribe(temp_file.name)
                os.unlink(temp_file.name)
                return result["text"].strip()
        except Exception as e:
            logging.error(f"Error with Whisper transcription: {e}")
            return "[ERROR]"
    
    def transcribe_chunk_sphinx(self, audio_chunk: AudioSegment) -> str:
        """Transcribe audio chunk using CMU Sphinx (offline)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_chunk.export(temp_file.name, format="wav")
                
                with sr.AudioFile(temp_file.name) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_sphinx(audio_data)
                
                os.unlink(temp_file.name)
                return text
        except sr.UnknownValueError:
            return "[INAUDIBLE]"
        except Exception as e:
            logging.error(f"Error with Sphinx transcription: {e}")
            return "[ERROR]"
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe entire audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        logging.info(f"Starting transcription using {self.engine} engine")
        
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Split audio into chunks
        chunks = self.split_audio_on_silence(audio)
        
        # Transcribe each chunk
        transcript_parts = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Transcribing chunk {i+1}/{len(chunks)}")
            
            if self.engine == "whisper" and WHISPER_AVAILABLE:
                text = self.transcribe_chunk_whisper(chunk)
            elif self.engine == "sphinx":
                text = self.transcribe_chunk_sphinx(chunk)
            else:  # Default to Google
                text = self.transcribe_chunk_google(chunk)
            
            if text and text not in ["[INAUDIBLE]", "[ERROR]"]:
                transcript_parts.append(text)
        
        return " ".join(transcript_parts)
    
    def transcribe_video(self, video_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Complete video transcription pipeline.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output text file (optional)
            
        Returns:
            Tuple[bool, str]: (Success status, transcript text or error message)
        """
        if not os.path.exists(video_path):
            return False, f"Video file not found: {video_path}"
        
        # Generate output path if not provided
        if not output_path:
            video_stem = Path(video_path).stem
            output_path = f"{video_stem}_transcript.txt"
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio
            if not self.extract_audio(video_path, temp_audio_path):
                return False, "Failed to extract audio from video"
            
            # Transcribe audio
            transcript = self.transcribe_audio(temp_audio_path)
            
            if not transcript.strip():
                return False, "No speech detected in audio"
            
            # Save transcript
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            logging.info(f"Transcript saved to {output_path}")
            return True, transcript
            
        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            logging.error(error_msg)
            return False, error_msg
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Transcribe video files to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_transcriber.py video.mp4
  python video_transcriber.py video.mp4 transcript.txt
  python video_transcriber.py video.mp4 -e whisper -m small
  python video_transcriber.py video.mp4 -e sphinx --verbose
        """
    )
    
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_text", nargs="?", help="Path to output text file (optional)")
    parser.add_argument("-e", "--engine", choices=["google", "whisper", "sphinx"], 
                       default="google", help="Transcription engine (default: google)")
    parser.add_argument("-m", "--model", default="base", 
                       help="Whisper model size (tiny/base/small/medium/large, default: base)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create transcriber
    transcriber = VideoTranscriber(engine=args.engine, model_size=args.model)
    
    # Transcribe video
    success, result = transcriber.transcribe_video(args.input_video, args.output_text)
    
    if success:
        print(f"✓ Transcription completed successfully!")
        if args.output_text:
            print(f"✓ Saved to: {args.output_text}")
        else:
            video_stem = Path(args.input_video).stem
            print(f"✓ Saved to: {video_stem}_transcript.txt")
        
        if args.verbose:
            print(f"\nTranscript preview:\n{result[:500]}...")
    else:
        print(f"✗ Transcription failed: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
""" 
# Basic usage (uses Google Speech Recognition)
python video_transcriber.py input_video.mp4

# Specify output file
python video_transcriber.py input_video.mp4 transcript.txt

# Use Whisper engine with small model
python video_transcriber.py input_video.mp4 -e whisper -m small

# Use offline Sphinx engine with verbose output
python video_transcriber.py input_video.mp4 -e sphinx --verbose
"""  