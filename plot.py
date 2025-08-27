from attacks import *
import matplotlib.pyplot as plt

def plot(fgsm_res, pgd_res, df_res, ber_clean_pct):
    """
    Encode watermark 32 bit payload into audio
    :param input: path to file original audio
    :param output: path to save file after watermarking
    :return: numpy array of data audio after watermarking
    """
    #FGSM & PGD
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(fgsm_res["eps"], fgsm_res["ber"], marker='o', linewidth=2, label=fgsm_res["name"])
    plt.semilogx(pgd_res["eps"],  pgd_res["ber"],  marker='s', linewidth=2, label=pgd_res["name"])
    plt.axhline(ber_clean_pct, linestyle='--', linewidth=1.5, label=f'Clean BER = {ber_clean_pct:.2f}%')
    plt.xlabel('Epsilon')
    plt.ylabel('BER (%)')
    plt.title('BER vs Epsilon: FGSM & PGD')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()

    #DeepFool
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(df_res["eps2"], df_res["ber"], marker='^', linewidth=2, label=df_res["name"])
    plt.axhline(ber_clean_pct, linestyle='--', linewidth=1.5, label=f'Clean BER = {ber_clean_pct:.2f}%')
    plt.xlabel('Epsilon (L2)')
    plt.ylabel('BER (%)')
    plt.title('BER vs ε Epsilon: DeepFool ')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Device:", DEVICE)

    payload_bits = np.array([
        0, 0, 0, 1, 1, 1, 0, 0,
        1, 0, 0, 1, 1, 0, 1, 1,
        0, 1, 0, 0, 1, 1, 1, 0,
        1, 1, 0, 1, 0, 0, 1, 1])

    wm_model, fmodel, x_batch, payload_t = load_model_and_batch(
        wav_path="wav/watermarks/output_1.wav",
        payload_bits=payload_bits
    )

    ber_clean = bit_error_rate(wm_model, x_batch, payload_t)
    print(f"Clean BER: {ber_clean * 100:.2f}%")

    print('------------FGSM-------------')
    fgsm_res = fgsm_attack(wm_model, fmodel, x_batch, payload_t)
    print('------------PGD--------------')
    pgd_res = pgd_attack(wm_model, fmodel, x_batch, payload_t)
    print('---------DeepFool------------')
    df_res = deepfool_attack(wm_model, x_batch, payload_t)

    plot(fgsm_res, pgd_res, df_res, ber_clean_pct=ber_clean * 100.0)
