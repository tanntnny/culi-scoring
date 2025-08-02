from transformers import Wav2Vec2Processor, Wav2Vec2Model

Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir="models/wav2vec2-processor")
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir="models/wav2vec2-model")