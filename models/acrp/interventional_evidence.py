import torch
from torch import nn


class InterventionalEvidenceEstimator(nn.Module):
    def __init__(self, hidden_dim=256, num_findings=10, temperature=0.5):
        super().__init__()
        self.finding_head = nn.Linear(hidden_dim, num_findings)
        self.temperature = temperature

    def _predict(self, anatomy_tokens):
        pooled = anatomy_tokens.mean(dim=1)
        return self.finding_head(pooled)

    def forward(self, anatomy_tokens):
        original_logits = self._predict(anatomy_tokens)
        original_probs = torch.sigmoid(original_logits)
        effects = []

        for region_id in range(anatomy_tokens.size(1)):
            intervened = anatomy_tokens.clone()
            intervened[:, region_id, :] = 0
            counterfactual_probs = torch.sigmoid(self._predict(intervened))
            effects.append(original_probs - counterfactual_probs)

        interventional_effects = torch.stack(effects, dim=1)
        positive_effect = torch.relu(interventional_effects).max(dim=1).values
        gate = torch.sigmoid(positive_effect / self.temperature)
        gated_scores = original_probs * gate
        return interventional_effects, gated_scores, original_logits
