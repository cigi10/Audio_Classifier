#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import time
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers

# audio config - 16khz sample rate, 0.5s chunks
sample_rate = 16000  
chunk_size = 8000
update_interval = 1.0
volume_threshold = 0.008
max_display = 3

class sounddetector:
    def __init__(self, model_path="yamnet.tflite"):
        # init audio classifier with optimized settings
        self.classifier = audio.audioclassifier.create_from_options(
            audio.audioclassifieroptions(
                base_options=python.baseoptions(model_asset_path=model_path),
                max_results=5,
                score_threshold=0.15
            )
        )
        
        # sound mapping with custom detection params
        self.sound_map = {
            "clap": {"icon": "👏", "matches": ["clap"]},
            "whistle": {
                "icon": "🎶", 
                "matches": ["whistle", "whistling"],
                "boost": 1.4,  # higher score boost
                "min_freq": 1000,  # freq range for whistle
                "max_freq": 4000   
            },
            # ... other sound mappings ...
        }

    def process_audio(self, audio_data):
        try:
            # first check for special cases
            if self._is_clap(audio_data):
                return [("clap", "👏", 0.85)]
                
            if self._is_whistle(audio_data):
                return [("whistle", "🎶", 0.9)]
            
            # normal classification flow
            audio_clip = containers.audiodata.create_from_array(
                audio_data.astype(np.float32),
                sample_rate=sample_rate
            )
            results = self.classifier.classify(audio_clip)
            
            # process classification results
            detected = []
            if results and results[0].classifications:
                for category in results[0].classifications[0].categories:
                    raw_name = category.category_name.split("_")[0].lower()
                    for name, data in self.sound_map.items():
                        if raw_name in data.get("matches", []):
                            score = min(1.0, category.score * data.get("boost", 1.0))
                            detected.append((name, data["icon"], score))
            
            return detected if detected else none
            
        except exception as e:
            print(f"error: {str(e)[:50]}")
            return none

    def _is_clap(self, audio_data):
        """check for sharp, brief impulse sounds"""
        peak = np.max(np.abs(audio_data))
        if peak < 0.3: return false
        return np.sum(np.abs(audio_data) > 0.2) < 100  # short duration

    def _is_whistle(self, audio_data):
        """specialized whistle detection using fft"""
        spectrum = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        
        # check energy in whistle range
        in_range = (freqs > 1000) & (freqs < 4000)
        whistle_energy = np.sum(spectrum[in_range])
        total_energy = np.sum(spectrum)
        
        # check tonal characteristics
        peaks = spectrum[np.argsort(spectrum)[-3:]]  # top 3 peaks
        harmonicity = np.std(peaks)/(np.mean(peaks)+1e-10)
        
        return (whistle_energy/total_energy > 0.7 and 
                harmonicity < 0.3 and
                np.max(audio_data) > 0.15)

def main():
    print("=== sound detector ===")
    detector = sounddetector()
    last_update = time.time()
    sound_buffer = defaultdict(list)
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal last_update
    
        # process incoming audio chunk
        audio_data = np.mean(indata, axis=1) / 32768.0
        current_volume = np.max(np.abs(audio_data))
        
        if current_volume > volume_threshold:
            detections = detector.process_audio(audio_data)
            if detections:
                for name, icon, score in detections:
                    sound_buffer[name].append(score)
        
        # timed output every interval
        if time.time() - last_update >= update_interval:
            if sound_buffer:
                # get top sounds by average score
                top_sounds = sorted(
                    [(name, np.mean(scores)) 
                     for name, scores in sound_buffer.items()],
                    key=lambda x: x[1], 
                    reverse=true
                )[:max_display]
                
                # format display string
                display = []
                for name, avg_score in top_sounds:
                    icon = detector.sound_map[name]["icon"]
                    display.append(f"{icon} {name}: {avg_score:.0%}")
                
                print("\r" + " | ".join(display) + " " * 20, end="")
                sound_buffer.clear()
            else:
                print("\rlistening...", end="")
            
            last_update = time.time()

    try:
        # start audio stream
        with sd.inputstream(
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            callback=audio_callback,
            dtype=np.int16
        ):
            print("ready (press ctrl+c to stop)")
            while true:
                sd.sleep(100)
    except keyboardinterrupt:
        print("\n\ndetection stopped")

if __name__ == "__main__":
    main()
