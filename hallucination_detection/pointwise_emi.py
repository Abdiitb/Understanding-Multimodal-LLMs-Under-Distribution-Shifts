from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class PointwiseEMIResult:
	pointwise_emi: torch.Tensor
	mi_true: torch.Tensor
	mi_model: torch.Tensor
	positive_log_q_true: torch.Tensor
	negative_log_q_true_mean: torch.Tensor
	positive_log_q_model: torch.Tensor
	negative_log_q_model_mean: torch.Tensor


class PointwiseEMI:
	"""
	Compute pointwise EMI from a trained CLUB estimator using:
	EMI_i = MI_true_i - MI_model_i

	For each sample i:
	- MI_true_i uses y_i^true and negatives sampled from true responses.
	- MI_model_i uses y_i^model and negatives sampled from model responses.
	"""

	def __init__(
		self,
		club_estimator: nn.Module,
		num_negative_samples: int = 50,
		device: torch.device | str | None = None,
		seed: int | None = None,
	) -> None:
		if num_negative_samples <= 0:
			raise ValueError("num_negative_samples must be > 0")

		self.club = club_estimator
		self.num_negative_samples = num_negative_samples
		self.device = torch.device(device) if device is not None else next(self.club.parameters()).device
		self.generator = torch.Generator(device="cpu")
		if seed is not None:
			self.generator.manual_seed(seed)

		self.club.eval()

	def log_q_endpoint(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
		"""
		CLUB endpoint for log q_theta(y | x), matching CLUB.loglikeli form.

		Args:
			x_samples: Tensor of shape [B, D]
			y_samples: Tensor of shape [B, D]
		Returns:
			Tensor of shape [B] with pointwise log-q scores.
		"""
		if not hasattr(self.club, "get_mu_logvar"):
			raise AttributeError("club_estimator must provide get_mu_logvar(x)")

		x_samples = x_samples.to(self.device)
		y_samples = y_samples.to(self.device)

		with torch.inference_mode():
			mu, logvar = self.club.get_mu_logvar(x_samples)
			return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1)

	def _sample_negative_indices(self, i: int, n: int) -> torch.Tensor:
		if n <= 1:
			raise ValueError("Need at least 2 samples to draw negatives with j != i")

		candidate_indices = torch.cat(
			[
				torch.arange(0, i, dtype=torch.long),
				torch.arange(i + 1, n, dtype=torch.long),
			]
		)

		if self.num_negative_samples <= (n - 1):
			perm = torch.randperm(n - 1, generator=self.generator)
			return candidate_indices[perm[: self.num_negative_samples]]

		sampled = torch.randint(0, n - 1, (self.num_negative_samples,), generator=self.generator)
		return candidate_indices[sampled]

	def compute(self, x: torch.Tensor, y_true: torch.Tensor, y_model: torch.Tensor) -> PointwiseEMIResult:
		if x.ndim != 2 or y_true.ndim != 2 or y_model.ndim != 2:
			raise ValueError("x, y_true, and y_model must be 2D tensors [N, D]")
		if x.shape != y_true.shape:
			raise ValueError(f"Shape mismatch: x={tuple(x.shape)} vs y_true={tuple(y_true.shape)}")
		if x.shape != y_model.shape:
			raise ValueError(f"Shape mismatch: x={tuple(x.shape)} vs y_model={tuple(y_model.shape)}")

		n = x.shape[0]
		if n <= 1:
			raise ValueError("Need at least 2 paired samples to compute pointwise EMI")

		x = x.to(self.device)
		y_true = y_true.to(self.device)
		y_model = y_model.to(self.device)

		positive_true_terms: list[torch.Tensor] = []
		negative_true_means: list[torch.Tensor] = []
		mi_true_terms: list[torch.Tensor] = []

		positive_model_terms: list[torch.Tensor] = []
		negative_model_means: list[torch.Tensor] = []
		mi_model_terms: list[torch.Tensor] = []

		emi_terms: list[torch.Tensor] = []

		for i in range(n):
			x_i = x[i : i + 1]

			# Step 1: MI for TRUE distribution
			y_i_true = y_true[i : i + 1]
			pos_true = self.log_q_endpoint(x_i, y_i_true).squeeze(0)

			neg_true_indices = self._sample_negative_indices(i=i, n=n)
			x_rep_true = x_i.expand(neg_true_indices.shape[0], -1)
			y_neg_true = y_true[neg_true_indices]
			neg_true = self.log_q_endpoint(x_rep_true, y_neg_true).mean()
			mi_true_i = pos_true - neg_true

			# Step 2: MI for MODEL distribution
			y_i_model = y_model[i : i + 1]
			pos_model = self.log_q_endpoint(x_i, y_i_model).squeeze(0)

			neg_model_indices = self._sample_negative_indices(i=i, n=n)
			x_rep_model = x_i.expand(neg_model_indices.shape[0], -1)
			y_neg_model = y_model[neg_model_indices]
			neg_model = self.log_q_endpoint(x_rep_model, y_neg_model).mean()
			mi_model_i = pos_model - neg_model

			# Step 3: EMI
			emi_i = mi_true_i - mi_model_i

			positive_true_terms.append(pos_true)
			negative_true_means.append(neg_true)
			mi_true_terms.append(mi_true_i)

			positive_model_terms.append(pos_model)
			negative_model_means.append(neg_model)
			mi_model_terms.append(mi_model_i)

			emi_terms.append(emi_i)

		return PointwiseEMIResult(
			pointwise_emi=torch.stack(emi_terms),
			mi_true=torch.stack(mi_true_terms),
			mi_model=torch.stack(mi_model_terms),
			positive_log_q_true=torch.stack(positive_true_terms),
			negative_log_q_true_mean=torch.stack(negative_true_means),
			positive_log_q_model=torch.stack(positive_model_terms),
			negative_log_q_model_mean=torch.stack(negative_model_means),
		)

	def compute_from_pairs(
		self,
		pairs_true: Sequence[tuple[torch.Tensor, torch.Tensor]],
		pairs_model: Sequence[tuple[torch.Tensor, torch.Tensor]],
	) -> PointwiseEMIResult:
		"""
		Convenience wrapper for two aligned pair lists:
		- pairs_true:  (x_i, y_i_true)
		- pairs_model: (x_i, y_i_model)
		"""
		if len(pairs_true) == 0 or len(pairs_model) == 0:
			raise ValueError("pairs_true and pairs_model must be non-empty")
		if len(pairs_true) != len(pairs_model):
			raise ValueError("pairs_true and pairs_model must have the same length")

		x_true = torch.stack([p[0] for p in pairs_true], dim=0)
		y_true = torch.stack([p[1] for p in pairs_true], dim=0)
		x_model = torch.stack([p[0] for p in pairs_model], dim=0)
		y_model = torch.stack([p[1] for p in pairs_model], dim=0)

		if not torch.allclose(x_true, x_model):
			raise ValueError("Input x values in pairs_true and pairs_model must be aligned and equal")

		return self.compute(x=x_true, y_true=y_true, y_model=y_model)

