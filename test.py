from watermark import payload, encode_watermark, decode_watermark, ber

if __name__ == "__main__":
    filename = 'wav/samples/test_5.wav'
    output = 'wav/watermarks/output_5.wav'
    watermarked_signal = encode_watermark(filename=filename, output=output)
    payload_decoded = decode_watermark(watermarked_signal=watermarked_signal)
    ber(payload=payload, payload_decoded=payload_decoded)