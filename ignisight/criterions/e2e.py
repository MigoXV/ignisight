import math
from dataclasses import dataclass

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@register_criterion("ignisight_e2e")
class IgnisightE2ECriterion(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        images, tgt_vectors = sample

        hyp_vectors = model(images)
        loss_vector = torch.nn.functional.mse_loss(
            hyp_vectors, tgt_vectors, reduction="sum" if reduce else "none"
        )
        sample_size = tgt_vectors.numel()
        ir_upper_loss = loss_vector[0]
        ir_left2_loss = loss_vector[1]
        ir_sic_upper_loss = loss_vector[2]
        left_2_loss = loss_vector[3]
        upper_loss = loss_vector[4]
        sic_upper_loss = loss_vector[5]
        loss = loss_vector.sum()

        logging_output = {
            "loss": loss.data,
            "ir_upper_loss": ir_upper_loss.data,
            "ir_left2_loss": ir_left2_loss.data,
            "ir_sic_upper_loss": ir_sic_upper_loss.data,
            "left_2_loss": left_2_loss.data,
            "upper_loss": upper_loss.data,
            "sic_upper_loss": sic_upper_loss.data,
            "ntokens": sample_size,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def logging_outputs_can_be_summed(self):
        return True

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ir_upper_loss_sum = sum(log.get("ir_upper_loss", 0) for log in logging_outputs)
        ir_left2_loss_sum = sum(log.get("ir_left2_loss", 0) for log in logging_outputs)
        ir_sic_upper_loss_sum = sum(
            log.get("ir_sic_upper_loss", 0) for log in logging_outputs
        )
        left_2_loss_sum = sum(log.get("left_2_loss", 0) for log in logging_outputs)
        upper_loss_sum = sum(log.get("upper_loss", 0) for log in logging_outputs)
        sic_upper_loss_sum = sum(
            log.get("sic_upper_loss", 0) for log in logging_outputs
        )

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(
            "ir_upper_loss", ir_upper_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "ir_left2_loss", ir_left2_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "ir_sic_upper_loss",
            ir_sic_upper_loss_sum / sample_size,
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "left_2_loss", left_2_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "upper_loss", upper_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "sic_upper_loss", sic_upper_loss_sum / sample_size, sample_size, round=3
        )
