#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import time
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers

# Optimized Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 8000  # 0.5s chunks
UPDATE_INTERVAL = 1.0
VOLUME_THRESHOLD = 0.008
MAX_DISPLAY = 3

class SoundDetector:
    def __init__(self, model_path="yamnet.tflite"):
        self.classifier = audio.AudioClassifier.create_from_options(
            audio.AudioClassifierOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                max_results=5,
                score_threshold=0.15
            )
        )
        
        # Enhanced sound mapping with whistling parameters
        self.sound_map = {
            "Clap": {"icon": "👏", "matches": ["Clap"]},
            "Whistle": {
                "icon": "🎶",
                "matches": ["Whistle", "Whistling"],
                "boost": 1.4,  # Higher boost for whistling
                "min_freq": 1000,  # Hz
                "max_freq": 4000   # Hz
            },
            "Speech": {"icon": "🗣️", "matches": ["Speech"]},
            "Dog": {"icon": "🐕", "matches": ["Dog", "Bark"]},
            "Cat": {"icon": "🐈", "matches": ["Cat", "Meow"]},
            "Music": {"icon": "🎵", "matches": ["Music"]}
        }

    def process_audio(self, audio_data):
        try:
            # Special clap detection
            if self._is_clap(audio_data):
                return [("Clap", "👏", 0.85)]
                
            # Special whistle detection
            if self._is_whistle(audio_data):
                return [("Whistle", "🎶", 0.9)]
            
            # Normal classification
            audio_clip = containers.AudioData.create_from_array(
                audio_data.astype(np.float32),
                sample_rate=SAMPLE_RATE
            )
            results = self.classifier.classify(audio_clip)
            
            detected = []
            if results and results[0].classifications:
                for category in results[0].classifications[0].categories:
                    raw_name = category.category_name.split("_")[0]
                    for name, data in self.sound_map.items():
                        if raw_name in data.get("matches", []):
                            score = min(1.0, category.score * data.get("boost", 1.0))
                            detected.append((name, data["icon"], score))
            
            return detected if detected else None
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            return None

    def _is_clap(self, audio_data):
        """Detect sharp, brief sounds"""
        peak = np.max(np.abs(audio_data))
        if peak < 0.3: return False
        return np.sum(np.abs(audio_data) > 0.2) < 100

    def _is_whistle(self, audio_data):
        """Specialized whistle detection"""
        # 1. Frequency analysis
        spectrum = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/SAMPLE_RATE)
        
        # 2. Check energy in whistle frequency range
        in_whistle_range = (freqs > 1000) & (freqs < 4000)
        whistle_energy = np.sum(spectrum[in_whistle_range])
        total_energy = np.sum(spectrum)
        
        # 3. Check for tonal characteristics
        peaks = spectrum[np.argsort(spectrum)[-3:]]  # Top 3 peaks
        harmonicity = np.std(peaks)/(np.mean(peaks)+1e-10)
        
        return (whistle_energy/total_energy > 0.7 and 
                harmonicity < 0.3 and
                np.max(audio_data) > 0.15)

def main():
    print("=== Enhanced Sound Detector ===")
    print("Now detects: 👏Claps 🎶Whistles 🐕Dogs 🐈Cats\n")
    
    detector = SoundDetector()
    last_update = time.time()
    sound_buffer = defaultdict(list)
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal last_update
    
        audio_data = np.mean(indata, axis=1) / 32768.0
        current_volume = np.max(np.abs(audio_data))
        
        if current_volume > VOLUME_THRESHOLD:
            detections = detector.process_audio(audio_data)
            if detections:
                for name, icon, score in detections:
                    sound_buffer[name].append(score)
        
        # Timed output
        if time.time() - last_update >= UPDATE_INTERVAL:
            if sound_buffer:
                top_sounds = sorted(
                    [(name, np.mean(scores)) 
                     for name, scores in sound_buffer.items()],
                    key=lambda x: x[1], 
                    reverse=True
                )[:MAX_DISPLAY]
                
                display = []
                for name, avg_score in top_sounds:
                    icon = detector.sound_map[name]["icon"]
                    display.append(f"{icon} {name}: {avg_score:.0%}")
                
                print("\r" + " | ".join(display) + " " * 20, end="")
                sound_buffer.clear()
            else:
                print("\rListening...", end="")
            
            last_update = time.time()

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype=np.int16
        ):
            print("Ready (Press Ctrl+C to stop)")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n\nDetection stopped")

if __name__ == "__main__":
    main()