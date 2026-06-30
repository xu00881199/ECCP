import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .anatomy_tokenizer import AnatomyEvidenceTokenizer
from .interventional_evidence import InterventionalEvidenceEstimator


class ECCPNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_anatomy_regions=7,
        num_findings=10,
        hidden_dim=256,
        num_heads=8,
        max_report_len=128,
        visual_backbone="simple",
        pretrained_backbone=False,
        freeze_backbone=False,
        max_views=2,
        use_view_embedding=True,
        use_phase_embedding=False,
        use_plan_prompt=True,
        num_lesion_classes=0,
        use_lesion_prompt=False,
        num_language_prior_tokens=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_anatomy_regions = num_anatomy_regions
        self.num_findings = num_findings
        self.max_report_len = max_report_len
        self.visual_backbone_name = visual_backbone
        self.max_views = max_views
        self.use_view_embedding = use_view_embedding
        self.use_phase_embedding = use_phase_embedding
        self.use_plan_prompt = use_plan_prompt
        self.num_lesion_classes = num_lesion_classes
        self.use_lesion_prompt = bool(use_lesion_prompt and num_lesion_classes > 0)
        self.num_language_prior_tokens = max(int(num_language_prior_tokens), 0)

        self.visual_backbone, backbone_out_dim = self._build_visual_backbone(
            visual_backbone, hidden_dim, pretrained_backbone
        )
        self.visual_projection = nn.Identity() if backbone_out_dim == hidden_dim else nn.Conv2d(backbone_out_dim, hidden_dim, 1)
        self.view_embedding = nn.Embedding(max_views, hidden_dim)
        self.phase_embedding = nn.Embedding(3, hidden_dim)
        if freeze_backbone:
            for param in self.visual_backbone.parameters():
                param.requires_grad = False
        self.anatomy_tokenizer = AnatomyEvidenceTokenizer(
            num_anatomy_regions=num_anatomy_regions,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.intervention = InterventionalEvidenceEstimator(
            hidden_dim=hidden_dim,
            num_findings=num_findings,
        )
        self.anatomy_finding_head = nn.Linear(hidden_dim, num_findings)
        self.lesion_head = nn.Linear(hidden_dim, num_lesion_classes) if num_lesion_classes > 0 else None
        self.lesion_prompt_projection = (
            nn.Sequential(
                nn.LayerNorm(num_lesion_classes),
                nn.Linear(num_lesion_classes, hidden_dim),
                nn.Tanh(),
            )
            if self.use_lesion_prompt
            else None
        )
        self.polarity_head = nn.Linear(hidden_dim, 3)
        self.finding_state_embedding = nn.Embedding(2, hidden_dim)
        self.finding_index_embedding = nn.Embedding(num_findings, hidden_dim)
        self.plan_norm = nn.LayerNorm(hidden_dim)
        self.language_prior_tokens = (
            nn.Parameter(torch.empty(self.num_language_prior_tokens, hidden_dim))
            if self.num_language_prior_tokens > 0
            else None
        )
        self.language_prior_norm = nn.LayerNorm(hidden_dim)
        if self.language_prior_tokens is not None:
            nn.init.normal_(self.language_prior_tokens, mean=0.0, std=0.02)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_report_len, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def _build_visual_backbone(self, visual_backbone, hidden_dim, pretrained_backbone):
        if visual_backbone == "simple":
            return (
                nn.Sequential(
                    nn.Conv2d(3, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ),
                hidden_dim,
            )
        if visual_backbone in ("resnet18", "resnet34"):
            weights = None
            if pretrained_backbone:
                try:
                    if visual_backbone == "resnet18":
                        weights = models.ResNet18_Weights.IMAGENET1K_V1
                    else:
                        weights = models.ResNet34_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            backbone = models.resnet18(weights=weights) if visual_backbone == "resnet18" else models.resnet34(weights=weights)
            features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
            return features, 512
        raise ValueError("Unsupported visual_backbone: %s" % visual_backbone)

    def encode_visual(self, images, view_phase_ids=None):
        if images.dim() == 5:
            batch_size, num_views = images.shape[:2]
            if num_views > self.max_views:
                raise ValueError("Received %s views, but max_views=%s" % (num_views, self.max_views))
            flat_images = images.reshape(batch_size * num_views, *images.shape[2:])
            feats = self.visual_backbone(flat_images)
            feats = self.visual_projection(feats)
            visual_tokens = feats.flatten(2).transpose(1, 2)
            tokens_per_view = visual_tokens.size(1)
            visual_tokens = visual_tokens.reshape(batch_size, num_views, tokens_per_view, -1)
            if self.use_view_embedding:
                view_ids = torch.arange(num_views, device=images.device).view(1, num_views, 1)
                visual_tokens = visual_tokens + self.view_embedding(view_ids)
            if self.use_phase_embedding:
                if view_phase_ids is None:
                    relative_positions = torch.linspace(0, 1, num_views, device=images.device)
                    phase_ids = torch.zeros(num_views, dtype=torch.long, device=images.device)
                    phase_ids = phase_ids.masked_fill(relative_positions >= 0.25, 1)
                    phase_ids = phase_ids.masked_fill(relative_positions >= 0.65, 2)
                    view_phase_ids = phase_ids.unsqueeze(0).expand(batch_size, -1)
                else:
                    view_phase_ids = view_phase_ids.to(device=images.device, dtype=torch.long).clamp(0, 2)
                visual_tokens = visual_tokens + self.phase_embedding(view_phase_ids).unsqueeze(2)
            visual_tokens = visual_tokens.reshape(batch_size, num_views * tokens_per_view, -1)
            anatomy_tokens, attention_maps = self.anatomy_tokenizer(visual_tokens)
            return anatomy_tokens, attention_maps
        feats = self.visual_backbone(images)
        feats = self.visual_projection(feats)
        visual_tokens = feats.flatten(2).transpose(1, 2)
        anatomy_tokens, attention_maps = self.anatomy_tokenizer(visual_tokens)
        return anatomy_tokens, attention_maps

    def encode_plan_prompt(self, plan_finding_prompt, batch_size, device):
        if plan_finding_prompt is None:
            plan_finding_prompt = torch.zeros(
                batch_size, self.num_findings, dtype=torch.long, device=device
            )
        else:
            plan_finding_prompt = plan_finding_prompt.to(device=device, dtype=torch.long).clamp(0, 1)
        finding_ids = torch.arange(self.num_findings, device=device).unsqueeze(0).expand(batch_size, -1)
        plan_tokens = self.finding_state_embedding(plan_finding_prompt) + self.finding_index_embedding(finding_ids)
        return self.plan_norm(plan_tokens)

    def encode_language_prior(self, batch_size, device):
        if self.language_prior_tokens is None:
            return None
        tokens = self.language_prior_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.language_prior_norm(tokens)

    def decode_with_memory(self, report_ids, decoder_memory):
        positions = torch.arange(report_ids.size(1), device=report_ids.device).unsqueeze(0)
        tgt = self.token_embedding(report_ids) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(report_ids.size(1)).to(report_ids.device)
        decoded = self.decoder(tgt=tgt, memory=decoder_memory, tgt_mask=causal_mask)
        return self.output_head(decoded)

    def forward_language_model(self, report_ids):
        prior_tokens = self.encode_language_prior(report_ids.size(0), report_ids.device)
        if prior_tokens is None:
            prior_tokens = self.token_embedding.weight.new_zeros(report_ids.size(0), 1, self.token_embedding.embedding_dim)
        logits = self.decode_with_memory(report_ids, prior_tokens)
        return {"logits": logits, "language_prior_tokens": prior_tokens}

    @torch.no_grad()
    def predict_plan_finding_prompt(self, images, threshold=0.5, view_phase_ids=None):
        anatomy_tokens, _ = self.encode_visual(images, view_phase_ids=view_phase_ids)
        _, _, entity_logits = self.intervention(anatomy_tokens)
        return (torch.sigmoid(entity_logits) > threshold).long()

    def forward(self, images, report_ids, plan_finding_prompt=None, view_phase_ids=None):
        anatomy_tokens, attention_maps = self.encode_visual(images, view_phase_ids=view_phase_ids)
        interventional_effects, gated_scores, entity_logits = self.intervention(anatomy_tokens)
        anatomy_finding_logits = self.anatomy_finding_head(anatomy_tokens)
        polarity_logits = self.polarity_head(anatomy_tokens.mean(dim=1))
        lesion_logits = self.lesion_head(anatomy_tokens.mean(dim=1)) if self.lesion_head is not None else None
        if self.use_plan_prompt:
            plan_tokens = self.encode_plan_prompt(plan_finding_prompt, images.size(0), images.device)
            memory_parts = [plan_tokens]
        else:
            plan_tokens = anatomy_tokens.new_zeros(images.size(0), 0, anatomy_tokens.size(-1))
            memory_parts = []
        language_prior = self.encode_language_prior(images.size(0), images.device)
        if language_prior is not None:
            memory_parts.append(language_prior)
        if self.lesion_prompt_projection is not None and lesion_logits is not None:
            lesion_prompt = self.lesion_prompt_projection(torch.sigmoid(lesion_logits)).unsqueeze(1)
            memory_parts.append(lesion_prompt)
        memory_parts.append(anatomy_tokens)
        decoder_memory = torch.cat(memory_parts, dim=1)

        logits = self.decode_with_memory(report_ids, decoder_memory)

        outputs = {
            "logits": logits,
            "entity_logits": entity_logits,
            "anatomy_finding_logits": anatomy_finding_logits,
            "polarity_logits": polarity_logits,
            "anatomy_tokens": anatomy_tokens,
            "plan_tokens": plan_tokens,
            "anatomy_attention_maps": attention_maps,
            "interventional_effects": interventional_effects,
            "gated_finding_scores": gated_scores,
        }
        if lesion_logits is not None:
            outputs["lesion_logits"] = lesion_logits
        return outputs

    def _apply_repetition_penalty(self, logits, generated_tokens, penalty):
        if penalty <= 1.0:
            return logits
        adjusted = logits.clone()
        for token_id in set(int(item) for item in generated_tokens.tolist()):
            if adjusted[token_id] < 0:
                adjusted[token_id] *= penalty
            else:
                adjusted[token_id] /= penalty
        return adjusted

    def _block_repeated_ngrams(self, logits, generated_tokens, no_repeat_ngram_size):
        if no_repeat_ngram_size <= 0 or generated_tokens.numel() < no_repeat_ngram_size - 1:
            return logits
        prefix = tuple(int(item) for item in generated_tokens[-(no_repeat_ngram_size - 1) :].tolist())
        banned = []
        token_list = [int(item) for item in generated_tokens.tolist()]
        for idx in range(len(token_list) - no_repeat_ngram_size + 1):
            ngram = tuple(token_list[idx : idx + no_repeat_ngram_size])
            if ngram[:-1] == prefix:
                banned.append(ngram[-1])
        if not banned:
            return logits
        adjusted = logits.clone()
        adjusted[torch.tensor(banned, dtype=torch.long, device=logits.device)] = -float("inf")
        return adjusted

    @torch.no_grad()
    def generate(
        self,
        images,
        start_token=1,
        end_token=2,
        max_len=128,
        plan_finding_prompt=None,
        num_beams=1,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        view_phase_ids=None,
    ):
        self.eval()
        batch_size = images.size(0)
        if self.use_plan_prompt and plan_finding_prompt is None:
            plan_finding_prompt = self.predict_plan_finding_prompt(images, view_phase_ids=view_phase_ids)
        elif not self.use_plan_prompt:
            plan_finding_prompt = None
        if num_beams > 1:
            return self._generate_beam_batched(
                images,
                plan_finding_prompt,
                start_token,
                end_token,
                max_len,
                num_beams,
                repetition_penalty,
                length_penalty,
                no_repeat_ngram_size,
                view_phase_ids,
            )
        generated = torch.zeros(batch_size, max_len, dtype=torch.long, device=images.device)
        generated[:, 0] = start_token
        for step in range(max_len - 1):
            outputs = self.forward(
                images,
                generated,
                plan_finding_prompt=plan_finding_prompt,
                view_phase_ids=view_phase_ids,
            )
            next_logits = outputs["logits"][:, step, :]
            if repetition_penalty > 1.0:
                next_logits = torch.stack(
                    [
                        self._apply_repetition_penalty(next_logits[idx], generated[idx, : step + 1], repetition_penalty)
                        for idx in range(batch_size)
                    ]
                )
            if no_repeat_ngram_size > 0:
                next_logits = torch.stack(
                    [
                        self._block_repeated_ngrams(next_logits[idx], generated[idx, : step + 1], no_repeat_ngram_size)
                        for idx in range(batch_size)
                    ]
                )
            next_token = next_logits.argmax(dim=-1)
            generated[:, step + 1] = next_token
            if torch.all(next_token == end_token):
                break
        return generated

    @torch.no_grad()
    def _generate_beam_batched(
        self,
        images,
        plan_finding_prompt,
        start_token,
        end_token,
        max_len,
        num_beams,
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
        view_phase_ids=None,
    ):
        batch_size = images.size(0)
        device = images.device
        vocab_size = self.vocab_size
        beam_images = (
            images.unsqueeze(1)
            .expand(batch_size, num_beams, *images.shape[1:])
            .reshape(batch_size * num_beams, *images.shape[1:])
        )
        beam_prompt = None
        if plan_finding_prompt is not None:
            beam_prompt = (
                plan_finding_prompt.unsqueeze(1)
                .expand(batch_size, num_beams, plan_finding_prompt.size(1))
                .reshape(batch_size * num_beams, plan_finding_prompt.size(1))
            )
        beam_phase_ids = None
        if view_phase_ids is not None:
            beam_phase_ids = (
                view_phase_ids.unsqueeze(1)
                .expand(batch_size, num_beams, view_phase_ids.size(1))
                .reshape(batch_size * num_beams, view_phase_ids.size(1))
            )
        sequences = torch.zeros(batch_size, num_beams, max_len, dtype=torch.long, device=device)
        sequences[:, :, 0] = start_token
        beam_scores = torch.full((batch_size, num_beams), -1.0e9, device=device)
        beam_scores[:, 0] = 0.0
        finished = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

        for step in range(max_len - 1):
            flat_sequences = sequences.reshape(batch_size * num_beams, max_len)
            outputs = self.forward(
                beam_images,
                flat_sequences,
                plan_finding_prompt=beam_prompt,
                view_phase_ids=beam_phase_ids,
            )
            next_logits = outputs["logits"][:, step, :]
            flat_finished = finished.reshape(batch_size * num_beams)

            adjusted_logits = []
            for idx in range(batch_size * num_beams):
                logits = next_logits[idx]
                if bool(flat_finished[idx].item()):
                    kept = torch.full_like(logits, -float("inf"))
                    kept[end_token] = 0.0
                    adjusted_logits.append(kept)
                    continue
                prefix = flat_sequences[idx, : step + 1]
                logits = self._apply_repetition_penalty(logits, prefix, repetition_penalty)
                logits = self._block_repeated_ngrams(logits, prefix, no_repeat_ngram_size)
                adjusted_logits.append(logits)
            next_logits = torch.stack(adjusted_logits, dim=0)

            log_probs = F.log_softmax(next_logits, dim=-1)
            raw_candidate_scores = (
                log_probs + beam_scores.reshape(batch_size * num_beams, 1)
            ).reshape(batch_size, num_beams * vocab_size)
            rank_denominator = max(float(step + 2) ** length_penalty, 1.0)
            ranked_scores = raw_candidate_scores / rank_denominator
            _, top_indices = torch.topk(ranked_scores, k=num_beams, dim=-1)
            next_beam_ids = top_indices // vocab_size
            next_token_ids = top_indices % vocab_size
            beam_scores = raw_candidate_scores.gather(1, top_indices)

            selected_sequences = sequences[batch_indices, next_beam_ids]
            sequences = selected_sequences.clone()
            sequences[:, :, step + 1] = next_token_ids
            selected_finished = finished[batch_indices, next_beam_ids]
            finished = selected_finished | (next_token_ids == end_token)
            if bool(finished.all().item()):
                break

        final_lengths = torch.full((batch_size, num_beams), max_len, dtype=torch.float32, device=device)
        eos_positions = sequences.eq(end_token)
        for batch_idx in range(batch_size):
            for beam_idx in range(num_beams):
                eos_idx = torch.nonzero(eos_positions[batch_idx, beam_idx], as_tuple=False)
                if eos_idx.numel() > 0:
                    final_lengths[batch_idx, beam_idx] = float(eos_idx[0].item() + 1)
        final_scores = beam_scores / torch.clamp(final_lengths.pow(length_penalty), min=1.0)
        best_beam_ids = final_scores.argmax(dim=-1)
        return sequences[torch.arange(batch_size, device=device), best_beam_ids]

    @torch.no_grad()
    def _generate_beam(
        self,
        images,
        plan_finding_prompt,
        start_token,
        end_token,
        max_len,
        num_beams,
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
    ):
        device = images.device
        results = []
        for batch_idx in range(images.size(0)):
            image = images[batch_idx : batch_idx + 1]
            prompt = None if plan_finding_prompt is None else plan_finding_prompt[batch_idx : batch_idx + 1]
            beams = [(torch.tensor([start_token], dtype=torch.long, device=device), 0.0, False)]
            for _ in range(max_len - 1):
                candidates = []
                for tokens, score, finished in beams:
                    if finished:
                        candidates.append((tokens, score, finished))
                        continue
                    decoder_input = F.pad(tokens, (0, max_len - tokens.numel())).unsqueeze(0)
                    outputs = self.forward(image, decoder_input, plan_finding_prompt=prompt)
                    next_logits = outputs["logits"][0, tokens.numel() - 1]
                    next_logits = self._apply_repetition_penalty(next_logits, tokens, repetition_penalty)
                    next_logits = self._block_repeated_ngrams(next_logits, tokens, no_repeat_ngram_size)
                    log_probs = F.log_softmax(next_logits, dim=-1)
                    top_scores, top_ids = torch.topk(log_probs, k=num_beams)
                    for token_score, token_id in zip(top_scores, top_ids):
                        next_tokens = torch.cat([tokens, token_id.view(1)])
                        candidates.append(
                            (
                                next_tokens,
                                score + float(token_score.item()),
                                int(token_id.item()) == end_token,
                            )
                        )
                beams = sorted(
                    candidates,
                    key=lambda item: item[1] / max(float(item[0].numel()) ** length_penalty, 1.0),
                    reverse=True,
                )[:num_beams]
                if all(item[2] for item in beams):
                    break
            best_tokens = beams[0][0]
            if best_tokens.numel() < max_len:
                best_tokens = F.pad(best_tokens, (0, max_len - best_tokens.numel()))
            results.append(best_tokens[:max_len])
        return torch.stack(results, dim=0)
