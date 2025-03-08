import os
from dataclasses import dataclass, field
from pathlib import Path

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
import sklearn.model_selection

from ignisight.datasets.e2e import IgnisightDataset
import pandas as pd
import sklearn


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
        self.df = pd.read_excel(config.train_xls_path)
        self.train_df, self.val_df = sklearn.model_selection.train_test_split(
            self.df, test_size=0.2
        )

    def load_dataset(self, split, **kwargs):

        if split == "train":
            self.datasets[split] = IgnisightDataset(self.train_image_dir, self.train_df)
        elif split == "valid":
            self.datasets[split] = IgnisightDataset(self.train_image_dir, self.val_df)
        else:
            raise KeyError(f"Invalid split: {split}")

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
