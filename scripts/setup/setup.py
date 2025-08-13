from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor.save_pretrained("models/wav2vec2-processor")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.save_pretrained("models/wav2vec2-model")
