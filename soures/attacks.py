import eagerpy as ep
import numpy as np
import torch
import wavmark
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD, LinfFastGradientAttack, L2DeepFoolAttack
from wavmark.utils import file_reader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
SAMPLES_PER_CHUNK = SR
BATCH_SIZE = 2
LINF_EPS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
L2_STEPS = 10
L2_BUDGETS = [0.001, 0.003, 0.005, 0.009, 0.01, 0.015, 0.02]


def load_model_and_batch(wav_path: str, payload_bits: np.ndarray):
    """
    Load model and batch from computer
    :param wav_path: path to file audio, which was watermarked
    :param payload_bits: payload
    :return: wavmark model, pytorch model, audio bath, paload
    """
    wm_model = wavmark.load_model().to(DEVICE).eval()
    sig_full = file_reader.read_as_single_channel(wav_path, aim_sr=SR).astype(np.float32)
    chunks = sig_full[:BATCH_SIZE * SAMPLES_PER_CHUNK].reshape(BATCH_SIZE, SAMPLES_PER_CHUNK)
    x_batch = torch.from_numpy(chunks).float().to(DEVICE)

    payload_t = torch.from_numpy(payload_bits).float().unsqueeze(0).to(DEVICE)

    class DecodeWrapper(torch.nn.Module):
        def __init__(self, wm):
            super().__init__()
            self.wm = wm
        def forward(self, x):
            return self.wm.decode(x)  # [B,32]

    fmodel = PyTorchModel(DecodeWrapper(wm_model).eval(), bounds=(-1, 1))
    return wm_model, fmodel, x_batch, payload_t


@torch.no_grad()
def bit_error_rate(model, x_wave, payload_bits):
    """
    Count bit error after decoding
    :param model: path to file original audio
    :param x_wave: path to save file after watermarking
    :param payload_bits: path to save file after watermarking
    :return: numpy array of data audio after watermarking
    """
    p = model.decode(x_wave).cpu().numpy() >= 0.5
    y = payload_bits.cpu().numpy().astype(np.int32)
    return float((p != y).mean())


class MatchLogitWrapper(torch.nn.Module):
    def __init__(self, wm, payload_bits):
        super().__init__()
        self.wm = wm
        self.register_buffer("ybits", payload_bits)
    def forward(self, x):
        p = self.wm.decode(x).clamp(1e-6, 1 - 1e-6)
        z = torch.log(p) - torch.log1p(-p)
        ypm1 = self.ybits * 2.0 - 1.0
        score = (z * ypm1).mean(dim=1, keepdim=True)
        return torch.cat([score, -score], dim=1)


@torch.no_grad()
def project_l2(x0, x, eps):
    delta = x - x0
    B = delta.shape[0]
    d2 = delta.view(B, -1)
    nrm = d2.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = (eps / nrm).clamp(max=1.0)
    return x0 + (d2 * factor).view_as(x)


def fgsm_attack(wm_model, fmodel, x_batch, payload_t):
    """
    FGSM attack over a list of epsilons and report BER for each epsilon.
    :param wm_model: The WavMark model used to decode bits and compute BER.
    :param fmodel: Foolbox-wrapped model (e.g., a DecodeWrapper) attacked by PGD.
    :param x_batch: Clean audio batch
    :param payload_t : Ground-truth payload bits of shape [1, 32] (float {0,1}).
    :return: dict 3 values name, epsilons, bit error rate for drawing plot
    """
    x = ep.astensor(x_batch)
    y = ep.astensor(torch.zeros(x_batch.shape[0], dtype=torch.long, device=DEVICE))
    eps_list, ber_list = [], []
    attack = LinfFastGradientAttack()
    for eps in LINF_EPS:
        _, advs, _ = attack(fmodel, x, y, epsilons=eps)
        ber = bit_error_rate(wm_model, advs.raw, payload_t)
        eps_list.append(eps)
        ber_list.append(ber * 100.0)
        print(f"\tEpsilon: {eps}, BER = {ber *100}%")
    return {"name": "FGSM ", "eps": eps_list, "ber": ber_list}


def pgd_attack(wm_model, fmodel, x_batch, payload_t):
    """
    Run an L-infinity PGD attack over a list of epsilons and report BER for each epsilon.
    :param wm_model: The WavMark model used to decode bits and compute BER.
    :param fmodel: Foolbox-wrapped model (e.g., a DecodeWrapper) attacked by PGD.
    :param x_batch: Clean audio batch
    :param payload_t : Ground-truth payload bits of shape [1, 32] (float {0,1}).
    :return: dict 3 values name, epsilons, bit error rate for drawing plot
    """
    x = ep.astensor(x_batch)
    y = ep.astensor(torch.zeros(x_batch.shape[0], dtype=torch.long, device=DEVICE))
    eps_list, ber_list = [], []
    attack = LinfPGD(steps=10, rel_stepsize=0.5, random_start=True)
    for eps in LINF_EPS:
        _, advs, _ = attack(fmodel, x, y, epsilons=eps)
        ber = bit_error_rate(wm_model, advs.raw, payload_t)
        eps_list.append(eps)
        ber_list.append(ber * 100.0)
        print(f"\tEpsilon: {eps}, BER = {ber * 100}%")
    return {"name": "PGD", "eps": eps_list, "ber": ber_list}


def deepfool_attack(wm_model, x_batch, payload_t):
    """
    Run an L-infinity PGD attack over a list of epsilons and report BER for each epsilon.
    :param wm_model: The WavMark model used to decode bits and compute BER.
    :param x_batch: Clean audio batch
    :param payload_t : Ground-truth payload bits of shape [1, 32] (float {0,1}).
    :return: dict 3 values name, epsilons, bit error rate for drawing plot
    """
    # DeepFool L2 trên classifier nhị phân match/mismatch
    clf = MatchLogitWrapper(wm_model, payload_t).eval()
    fmodel_df = PyTorchModel(clf, bounds=(-1, 1))

    x = ep.astensor(x_batch)
    y_cls = ep.astensor(torch.zeros(x_batch.shape[0], dtype=torch.long, device=DEVICE))

    df = L2DeepFoolAttack(steps=L2_STEPS)
    _, advs_df, _ = df(fmodel_df, x, y_cls, epsilons=None)
    advs_df.raw.detach().clone().to(x_batch.device)
    ber_curve = []
    for eps2 in L2_BUDGETS:
        adv_proj = project_l2(x_batch, advs_df.raw.detach().clone().to(x_batch.device), eps2).clamp(-1, 1)
        ber = bit_error_rate(wm_model, adv_proj, payload_t)
        ber_curve.append(ber * 100.0)
        print(f"\tEpsilon: {eps2}, BER = {ber * 100}%")
    return {"name": "DeepFool", "eps2": L2_BUDGETS, "ber": ber_curve}
