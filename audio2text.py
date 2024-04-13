from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
        device = "cuda"
else:
    device = "cpu"


tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

model = model.to(device)
speech, sample_rate = torchaudio.load("/Users/maximilianruess/Downloads/roy_keane.wav")

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    speech = resampler(speech)
    sample_rate = 16000
    
speech = torch.from_numpy(speech.squeeze().numpy()).to(device)

input_values = tokenizer(speech.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

input_values = input_values.to(device)

logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

print(transcription)