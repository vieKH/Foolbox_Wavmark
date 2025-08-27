import  numpy as np
import soundfile
import torch
import wavmark
from wavmark.utils import file_reader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)
payload = np.random.choice([0, 1], size=32)
payload_tensor = torch.from_numpy(payload).float().to(device).unsqueeze(0)


def encode_watermark(input: str, output: str) -> np.ndarray:
    """
    Encode watermark 32 bit payload into audio
    :param input: path to file original audio
    :param output: path to save file after watermarking
    :return: numpy array of data audio after watermarking
    """
    signal = file_reader.read_as_single_channel(input, aim_sr=16000).astype("float32")
    n_chunks = len(signal) // 16000
    signal = signal[: n_chunks * 16000]
    chunks = signal.reshape(n_chunks, 16000)
    encoded_chunks = []
    with torch.no_grad():
        for i in range(n_chunks):
            sig = torch.from_numpy(chunks[i]).float().to(device).unsqueeze(0)
            sig_wm = model.encode(sig, payload_tensor)
            encoded_chunks.append(sig_wm.cpu().numpy().squeeze())

    watermarked_signal = np.concatenate(encoded_chunks)
    soundfile.write(output, watermarked_signal, 16000)
    return watermarked_signal


def ber(payload: np.ndarray, payload_decoded: np.ndarray) -> None:
    """
    Count how many percent bit error
    :param payload: payload watermark into audio
    :param payload_decoded: payload after decoding
    :return: None
    """
    BER = (payload != payload_decoded).mean() * 100
    print("Decode BER: %.1f" % BER)


def decode_watermark(payload: np.ndarray, filename: str, check: bool) -> np.ndarray:
    """
    Decode audio for getting payload
    :param payload: payload was watermarked into audio, which use for counting bit error
    :param filename: payload after decoding
    :param check: param for counting bit error or not
    :return: payload after decoding
    """
    watermarked_signal = file_reader.read_as_single_channel(filename, aim_sr=16000).astype("float32")
    n_chunks = len(watermarked_signal) // 16000
    with torch.no_grad():
        sig = torch.from_numpy(watermarked_signal).float().to(device).reshape(n_chunks, 16000)
        decoded = (model.decode(sig) >= 0.5).int().cpu().numpy()

    if check:
        ber(payload=payload, payload_decoded=decoded)
    return decoded


if __name__ == "__main__":
    print("Payload: ", payload)
    input = '../wav/samples/test_1.wav'
    output = '../wav/watermarks/output_1.wav'
    watermarked_signal = encode_watermark(input=input, output=output)
    payload_decoded = decode_watermark(filename=output, payload=payload, check=True)
    print(payload_decoded)
