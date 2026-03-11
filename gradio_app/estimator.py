"""
CLUB mutual-information estimator — training and inference utilities.

Adapted from the main.py CLUB class.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from transformers import XLMRobertaModel, XLMRobertaTokenizer


# ---------------------------------------------------------------------------
# CLUB estimator (from https://github.com/Linear95/CLUB)
# ---------------------------------------------------------------------------
class CLUB(nn.Module):
    """Mutual Information Contrastive Learning Upper Bound estimator."""

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = -(mu - y_samples) ** 2 / 2.0 / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


# ---------------------------------------------------------------------------
# JSD helpers (from main.py)
# ---------------------------------------------------------------------------
def vonNeumannEntropy(K):
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]
    mek = mek / mek.sum()
    H = -1 * torch.sum(mek * torch.log(mek))
    return H


def JSD_cov(covX, covY):
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX + covY) / 2)
    return Hz - 0.5 * (Hx + Hy)


# ---------------------------------------------------------------------------
# Embedder wrapper
# ---------------------------------------------------------------------------
class Embedder:
    """Wraps CLIP (vision) + XLM-RoBERTa (text) encoders for embedding extraction."""

    def __init__(
        self,
        v_embedder_name: str = "openai/clip-vit-base-patch32",
        t_embedder_name: str = "xlm-roberta-base",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.v_model = CLIPModel.from_pretrained(v_embedder_name).to(self.device)
        self.v_processor = CLIPProcessor.from_pretrained(v_embedder_name)
        self.t_model = XLMRobertaModel.from_pretrained(t_embedder_name).to(self.device)
        self.t_tokenizer = XLMRobertaTokenizer.from_pretrained(t_embedder_name)

    @torch.inference_mode()
    def encode(self, images, questions, model_answers, ref_answers):
        """
        Encode a batch and return (z_v, z_t, z_yhat, z_y) all normalised 2-D tensors.
        """
        # Vision
        v_inputs = self.v_processor(images=images, return_tensors="pt", padding=True)
        v_inputs = {k: v.to(self.device) for k, v in v_inputs.items()}
        z_v = (
            self.v_model.vision_model(pixel_values=v_inputs["pixel_values"], output_hidden_states=True)
            .hidden_states[-1][:, 1:, :]
            .mean(dim=1)
            .float()
        )

        # Text inputs (questions)
        t_in = self.t_tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        t_in = {k: v.to(self.device) for k, v in t_in.items()}
        mask_i = t_in["attention_mask"].unsqueeze(-1)
        z_t = self.t_model(**t_in, output_hidden_states=True).hidden_states[0] * mask_i
        z_t = z_t[:, 1:, :].sum(dim=1) / mask_i[:, 1:, :].sum(dim=1)

        # Model answers
        t_yh = self.t_tokenizer(model_answers, return_tensors="pt", padding=True, truncation=True, max_length=512)
        t_yh = {k: v.to(self.device) for k, v in t_yh.items()}
        mask_yh = t_yh["attention_mask"].unsqueeze(-1)
        z_yhat = self.t_model(**t_yh, output_hidden_states=True).hidden_states[0] * mask_yh
        z_yhat = z_yhat[:, 1:, :].sum(dim=1) / mask_yh[:, 1:, :].sum(dim=1)

        # Reference answers
        t_y = self.t_tokenizer(ref_answers, return_tensors="pt", padding=True, truncation=True, max_length=512)
        t_y = {k: v.to(self.device) for k, v in t_y.items()}
        mask_y = t_y["attention_mask"].unsqueeze(-1)
        z_y = self.t_model(**t_y, output_hidden_states=True).hidden_states[0] * mask_y
        z_y = z_y[:, 1:, :].sum(dim=1) / mask_y[:, 1:, :].sum(dim=1)

        # Normalise
        z_v = F.normalize(z_v, p=2, dim=-1)
        z_t = F.normalize(z_t, p=2, dim=-1)
        z_yhat = F.normalize(z_yhat, p=2, dim=-1)
        z_y = F.normalize(z_y, p=2, dim=-1)

        return z_v, z_t, z_yhat, z_y


# ---------------------------------------------------------------------------
# CLUB training
# ---------------------------------------------------------------------------
def train_club(
    club: CLUB,
    embedder: Embedder,
    datasets_dict: dict,
    epochs: int = 500,
    lr: float = 1e-4,
    progress_callback=None,
):
    """
    Train the CLUB estimator on embeddings from all provided datasets.

    Args:
        club: uninitialised or partially-trained CLUB module.
        embedder: Embedder used to get (z_v, z_t, z_yhat, z_y) per split.
        datasets_dict: {split_name: hf_dataset_split} — each split should have
                        columns image, question, reference_answer.
                        model_answers are replaced with reference_answer for training.
        epochs: number of training epochs.
        lr: learning rate.
        progress_callback: optional callable(epoch, loss) for UI updates.
    Returns:
        trained CLUB module (same object, mutated in-place).
    """
    device = embedder.device
    club = club.to(device)
    optimizer = torch.optim.Adam(club.parameters(), lr=lr)

    # Pre-extract embeddings for all splits (use ref answers as model answers during training)
    all_z, all_y = [], []
    for split_name, ds in datasets_dict.items():
        images = list(ds["image"])
        questions = list(ds["question"])
        ref_answers = list(ds["reference_answer"])
        z_v, z_t, _, z_y = embedder.encode(images, questions, ref_answers, ref_answers)
        z = (z_v + z_t) * 0.5
        all_z.append(z)
        all_y.append(z_y)

    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)

    club.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = club.learning_loss(all_z, all_y)
        loss.backward()
        optimizer.step()
        if progress_callback and (epoch + 1) % 50 == 0:
            progress_callback(epoch + 1, loss.item())

    club.eval()
    return club


def load_club_checkpoint(ckpt_path: str, feature_dim: int = 768, hidden_dim: int = 500):
    """Load a pre-trained CLUB checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    club = CLUB(feature_dim, feature_dim, hidden_dim).to(device)
    club.load_state_dict(torch.load(ckpt_path, map_location=device))
    club.eval()
    return club


# ---------------------------------------------------------------------------
# EMI / EMID / EMID_UB computation
# ---------------------------------------------------------------------------
def compute_emi(club: CLUB, z_v, z_t, z_yhat, z_y):
    """Compute EMI = MI(z, y_hat) - MI(z, y) where z = (z_v + z_t) / 2."""
    device = next(club.parameters()).device
    z = ((z_v + z_t) * 0.5).to(device)
    z_yhat = z_yhat.to(device)
    z_y = z_y.to(device)
    with torch.inference_mode():
        model_mi = club(z, z_yhat).item()
        ref_mi = club(z, z_y).item()
    return model_mi - ref_mi, model_mi, ref_mi


def compute_emid(src_emi: float, tar_emi: float):
    """EMID = source EMI - target EMI."""
    return src_emi - tar_emi


def compute_emid_upperbound(p_zv, p_zt, p_zyh, p_zy, q_zv, q_zt, q_zyh, q_zy):
    """Compute the scale-adjusted EMID upper bound."""
    covPXv = torch.matmul(p_zv.T, p_zv)
    covQXv = torch.matmul(q_zv.T, q_zv)
    jsd_v = JSD_cov(covPXv, covQXv).item()

    covPXt = torch.matmul(p_zt.T, p_zt)
    covQXt = torch.matmul(q_zt.T, q_zt)
    jsd_t = JSD_cov(covPXt, covQXt).item()

    covPYH = torch.matmul(p_zyh.T, p_zyh)
    covPY = torch.matmul(p_zy.T, p_zy)
    jsd_py = JSD_cov(covPYH, covPY).item()

    covQYH = torch.matmul(q_zyh.T, q_zyh)
    covQY = torch.matmul(q_zy.T, q_zy)
    jsd_qy = JSD_cov(covQYH, covQY).item()

    emid_ub = jsd_v ** (1 / 2) + jsd_t ** (1 / 2) + jsd_py ** (1 / 4) + jsd_qy ** (1 / 4)
    return emid_ub
