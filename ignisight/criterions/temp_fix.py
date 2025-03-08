import torch
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("temp_fix")
class TempFixCriterion(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample):
        images, temps = sample
        B, C, H, W = images.shape
        ntokens = B * H * W
        hyp_temps = model(images)
        loss = torch.nn.functional.l1_loss(hyp_temps, temps, reduction="none")
        loss_avg = loss.mean()
        sample_size = images.size(0)
        logging_output = {
            "loss": loss.sum().item(),
            "ntokens": ntokens,
            "nsentences": B,
            "sample_size": B,
        }
        return loss_avg, sample_size, logging_output

    def logging_outputs_can_be_summed(self):
        return False

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum*1000 / ntokens, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens, sample_size, round=3)
        metrics.log_scalar("sample_size", sample_size, sample_size, round=3)
