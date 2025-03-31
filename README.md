
# Audio_Classifier  

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FF6F00?logo=google&logoColor=white)](https://mediapipe.dev)
[![Real-Time](https://img.shields.io/badge/Real_Time-✅-brightgreen)]()

---
### Sound Classifier's Core Capabilities:
- 500ms end-to-end latency (16kHz @ 8000-sample chunks)
- Signal processing: Custom FFT (whistles) + peak detection (claps)
- ML classification: MediaPipe's YAMNet (521-class model)
---

```mermaid
graph LR
A[Microphone] -->|16kHz chunks| B[Custom Preprocess]
B --> C{Is it a clap/whistle?}
C -->|Yes| D[ Fast-Track]
C -->|No| E[YAMNet ML Model]
D & E --> F[ Classified Result]
```
