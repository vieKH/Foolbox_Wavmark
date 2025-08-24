#I took this code in tutorial https://pypi.org/project/wavmark/
#A little change in line 17, tried to record my voice so use this func for transforming signal channel
import  numpy as np
import soundfile
import torch
import wavmark
from wavmark.utils import file_reader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)
payload = np.random.choice([0, 1], size=16)
print("Payload: ", payload)


def encode_watermark(filename: str, output: str):
    signal = file_reader.read_as_single_channel(filename, aim_sr=16000)
    watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)
    soundfile.write(output, watermarked_signal, 16000)
    return watermarked_signal


def decode_watermark(watermarked_signal: np.ndarray):
    payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
    print("Payload after decoding:: ", payload_decoded)
    return payload_decoded


def ber(payload: np.ndarray, payload_decoded: np.ndarray):
    BER = (payload != payload_decoded).mean() * 100
    print("Decode BER: %.1f" % BER)


if __name__ == "__main__":
    filename = 'wav/samples/test_2.wav'
    output = 'wav/watermarks/output_2.wav'
    watermarked_signal = encode_watermark(filename=filename, output=output)
    payload_decoded = decode_watermark(watermarked_signal=watermarked_signal)
    ber(payload=payload, payload_decoded=payload_decoded)