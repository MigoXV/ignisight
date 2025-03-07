import os
from dataclasses import dataclass, field
from pathlib import Path

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from ignisight.datasets.e2e import IgnisightDataset


@dataclass
class IgnisightConfig(FairseqDataclass):
    train_image_dir: str = field(
        # default="data/e2e/images",
        default=os.getenv("IGNISIGHT_IMAGE_DIR"),
        metadata={"help": "path to image directory"},
    )
    train_xls_path: str = field(
        # default="data/e2e/data.xlsx",
        default=os.getenv("IGNISIGHT_XLS_PATH"),
        metadata={"help": "path to xls file"},
    )


@register_task("ignisight_e2e", dataclass=IgnisightConfig)
class IgnisightE2ETask(FairseqTask):

    def __init__(self, config: IgnisightConfig):
        super().__init__(config)
        self.train_image_dir = Path(config.train_image_dir)
        self.train_xls_path = Path(config.train_xls_path)

    def load_dataset(self, split, **kwargs):

        if split == "train":
            self.datasets[split] = IgnisightDataset(
                self.train_image_dir, self.train_xls_path
            )
        elif split == "valid":
            self.datasets[split] = IgnisightDataset(
                self.train_image_dir, self.train_xls_path
            )
        else:
            raise KeyError(f"Invalid split: {split}")

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
