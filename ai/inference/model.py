"""
PoseT5: T5 model with pose keypoint encoder for Signova AI (F-013).
Official architecture from manavdhamecha77/iSign-t5-pose-to-text.
Maps pose keypoints -> T5 embedding space -> T5 encoder-decoder.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


class PoseEncoder(nn.Module):
    """
    Encodes pose keypoints into T5 embedding space.

    Input: (batch_size, seq_length, pose_dim)
    Output: (batch_size, seq_length, embedding_dim)
    """

    def __init__(self, pose_dim: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.pose_dim = pose_dim
        self.embedding_dim = embedding_dim

        # Simple MLP: pose_dim -> hidden_dim -> embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (batch_size, seq_length, pose_dim)

        Returns:
            embeddings: (batch_size, seq_length, embedding_dim)
        """
        return self.encoder(pose)


class PoseT5(nn.Module):
    """
    T5 model with pose keypoint input.

    Architecture:
    - PoseEncoder: pose keypoints -> T5 embeddings
    - T5ForConditionalGeneration: standard encoder-decoder
    """

    def __init__(
        self,
        model_name: str,
        pose_dim: int,
        local_model_path: Optional[str] = None,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pose_dim = pose_dim

        # Load T5 model
        if local_model_path:
            logger.info("Loading T5 from local path: %s", local_model_path)
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                local_model_path,
                local_files_only=local_files_only,
            )
        else:
            logger.info("Loading T5 from HuggingFace: %s", model_name)
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )

        # Get embedding dimension from T5
        embedding_dim = self.t5.config.d_model

        # Create pose encoder
        self.pose_encoder = PoseEncoder(pose_dim, embedding_dim)

        logger.info(
            "PoseT5 initialized: pose_dim=%d, embedding_dim=%d",
            pose_dim,
            embedding_dim,
        )

    def forward(
        self,
        pose: torch.Tensor,
        pose_length: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass for training or loss evaluation."""
        pose_embeddings = self.pose_encoder(pose)

        batch_size, seq_length, _ = pose_embeddings.shape
        device = pose_embeddings.device

        pose_attention_mask = (
            torch.arange(seq_length, device=device)[None, :] < pose_length[:, None]
        ).long()

        outputs = self.t5(
            inputs_embeds=pose_embeddings,
            attention_mask=pose_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        pose: torch.Tensor,
        pose_length: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
        length_penalty: float = 2.0,
    ):
        """Generate text from pose input."""
        pose_embeddings = self.pose_encoder(pose)

        batch_size, seq_length, _ = pose_embeddings.shape
        device = pose_embeddings.device

        pose_attention_mask = (
            torch.arange(seq_length, device=device)[None, :] < pose_length[:, None]
        ).long()

        generated_ids = self.t5.generate(
            inputs_embeds=pose_embeddings,
            attention_mask=pose_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )

        return generated_ids

    def save_pretrained(self, save_path: str):
        """Save model and encoder."""
        os.makedirs(save_path, exist_ok=True)
        self.t5.save_pretrained(os.path.join(save_path, "t5"))
        torch.save(
            self.pose_encoder.state_dict(),
            os.path.join(save_path, "pose_encoder.pt"),
        )
        logger.info("Model saved to %s", save_path)

    def load_pretrained(
        self,
        load_path: str,
        load_pose_encoder: bool = True,
        load_t5: bool = True,
        local_files_only: bool = True,
    ):
        """Load T5 and pose encoder weights from checkpoint directory."""
        if load_t5:
            t5_path = os.path.join(load_path, "t5")
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                t5_path,
                local_files_only=local_files_only,
            )

        if load_pose_encoder:
            pose_encoder_path = os.path.join(load_path, "pose_encoder.pt")
            self.pose_encoder.load_state_dict(
                torch.load(
                    pose_encoder_path,
                    map_location="cpu",
                )
            )

        logger.info("Model loaded from %s", load_path)


def create_model(
    model_name: str,
    pose_dim: int,
    local_model_path: Optional[str] = None,
    local_files_only: bool = True,
) -> PoseT5:
    """Factory function to create PoseT5 model."""
    return PoseT5(
        model_name=model_name,
        pose_dim=pose_dim,
        local_model_path=local_model_path,
        local_files_only=local_files_only,
    )
