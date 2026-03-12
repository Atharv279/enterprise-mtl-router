import os
from moviepy.editor import VideoFileClip
from scenedetect import detect, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from paddleocr import PaddleOCR
import cv2

class LocalMediaProcessor:
    def __init__(self, output_dir="../data/processed/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize lightweight localized OCR; use_angle_cls=True helps with skewed text
        print("Initializing localized PaddleOCR engine...")
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def extract_audio_from_video(self, video_path: str) -> str:
        """Decouples audio without invoking heavy external shell subprocesses."""
        base_name = os.path.basename(video_path).split('.')[0]
        output_wav = os.path.join(self.output_dir, f"{base_name}.wav")
        
        # Extract and write to 16kHz WAV format natively
        print(f"Decoupling audio from {video_path}...")
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_wav, fps=16000, codec='pcm_s16le', logger=None)
        
        # Free memory immediately
        video.close()
        audio.close()
        
        return output_wav

    def extract_visual_text(self, video_path: str) -> str:
        """Detects scenes, extracts midpoint keyframes, and runs local OCR."""
        print(f"Running visual scene detection on {video_path}...")
        
        # 1. Scene Detection
        scene_list = detect(video_path, AdaptiveDetector())
        aggregated_visual_text = ""
        
        cap = cv2.VideoCapture(video_path)
        
        # 2. Extract midpoint keyframes and process
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            midpoint = int((start_frame + end_frame) / 2)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
            ret, frame = cap.read()
            
            if ret:
                # 3. Run localized OCR on the keyframe
                result = self.ocr_engine.ocr(frame, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.85: # Filter out low-confidence hallucinated text
                            aggregated_visual_text += f" [VISUAL_EVIDENCE: {text}] "
                            
        cap.release()
        return aggregated_visual_text.strip()