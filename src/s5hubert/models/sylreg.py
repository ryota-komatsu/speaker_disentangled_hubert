from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel

from .mincut import mincut_torch
from .modules import fix_random_seed


class SylRegForSyllableDiscovery(PreTrainedModel):
    config_class = HubertConfig
    base_model_prefix = "model"
    main_input_name = "input_values"

    def __init__(
        self,
        config,
        segmentation_layer: int = 6,
        n_units_step1: int = 24576,
        seed: int = 0,
        deduplicate: bool = True,
        sec_per_syllable: float = 0.15,
        merge_threshold: Optional[float] = 0.95,
        min_duration: int = 3,
        max_duration: int = 35,
    ):
        """
        Args:
            sec_per_syllable (`float`):
                Seconds per syllable, used to predefine the number of syllables in the input speech.
            merge_threshold (`float`, *optional*):
                Merge threshold of the cosine similarity between adjacent syllabic segments.
            min_duration (`int`):
                The minimum unit duration, measured in frames.
            max_duration (`int`):
                The maximum unit duration, measured in frames, before adjacent segment merge.
        """
        super().__init__(config)
        self.segmentation_layer = segmentation_layer
        self.deduplicate = deduplicate
        self.sec_per_frame = 0.02
        self.sec_per_syllable = sec_per_syllable
        self.merge_threshold = merge_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.model = HubertModel(config)
        self.model.eval()

        self.register_buffer("quantizer1", torch.rand(n_units_step1, config.hidden_size))
        self.register_buffer("quantizer2", torch.zeros(n_units_step1, dtype=torch.int))

        fix_random_seed(seed)

        self.post_init()

    @classmethod
    def load_pretrained(cls, model_path, quantizer1_path, quantizer2_path, **kwargs) -> "SylRegForSyllableDiscovery":
        """
        model = SylRegForSyllableDiscovery.load_pretrained(
            "models/sylreg",
            "models/sylreg/quantizer1.npy",
            "models/sylreg/quantizer2.npy",
        )
        model.push_to_hub("SylReg", private=True)
        """
        model = cls.from_pretrained(model_path, **kwargs)
        model.quantizer1 = torch.from_numpy(np.load(quantizer1_path))
        model.quantizer2 = torch.from_numpy(np.load(quantizer2_path))
        return model

    @torch.inference_mode()
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Raw speech waveform.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                1: non-padding
                0: padding

        Returns:
            units (`torch.LongTensor`):
                Discrete pseudo-syllabic units.
            durations (`torch.LongTensor`):
                Durations of units, measured in frames.
            dense (`torch.FloatTensor` of shape `((sequence_length - 400) // 320 + 1, hidden_size)`):
                Latent speech frame representations extracted from the syllable segmentation layer.
        """
        outputs = []

        hidden_states, padding_mask = self.model(input_values, attention_mask)
        hidden_states = hidden_states[self.segmentation_layer]
        lengths = (
            padding_mask.sum(dim=1)
            if padding_mask is not None
            else torch.full((hidden_states.size(0),), hidden_states.size(1), device=hidden_states.device)
        )

        batch_segments, batch_segment_features, batch_frame_boundary = mincut_torch(
            hidden_states,
            lengths,
            sec_per_frame=self.sec_per_frame,
            sec_per_syllable=self.sec_per_syllable,
            merge_threshold=self.merge_threshold,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            norm=True,
        )

        for dense, length, segments, segment_features, frame_boundary in zip(
            hidden_states, lengths, batch_segments, batch_segment_features, batch_frame_boundary
        ):
            dense = dense[:length]

            # Agglomerative clustering on K-means centroids
            units = self.quantizer2[torch.cdist(segment_features, self.quantizer1).argmin(1)]

            # deduplicate
            diff = units[1:] != units[:-1]
            start_mask = torch.cat([torch.tensor([True], device=units.device), diff])
            end_mask = torch.cat([diff, torch.tensor([True], device=units.device)])

            units = units[start_mask]
            frame_boundary = torch.stack([frame_boundary[:, 0][start_mask], frame_boundary[:, 1][end_mask]], dim=1)
            durations = frame_boundary[:, 1] - frame_boundary[:, 0]
            segments = frame_boundary * self.sec_per_frame

            segment_features = torch.segment_reduce(dense, "mean", lengths=durations)
            segment_features = (segment_features - segment_features.mean(dim=1, keepdim=True)) / segment_features.std(
                dim=1, keepdim=True
            )

            if not self.deduplicate:
                units = torch.repeat_interleave(units, durations)

            outputs.append(
                {
                    "units": units,
                    "durations": durations,
                    "dense": dense,
                    "segments": segments,
                    "segment_features": segment_features,
                }
            )
        return outputs
