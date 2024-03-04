from faster_whisper import WhisperModel
model_size = 'large-v3'
model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root='/usr/local')
