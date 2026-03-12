from faster_whisper import WhisperModel

class LocalTranscriptionEngine:
    def __init__(self, model_size="large-v3", device="cpu"):
        print(f"Loading {model_size} ASR model into {device} (int8 quantization)...")
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="int8" 
        )
        
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes audio and utilizes Silero VAD to trim silences.
        """
        # vad_filter=True invokes Silero VAD
        segments, info = self.model.transcribe(
            audio_path, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
            
        return transcribed_text.strip()