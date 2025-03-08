import os
from dataclasses import dataclass, field
from pathlib import Path

import sklearn
import sklearn.model_selection
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from ignisight.datasets.temp_fix import TempFixDataset


@dataclass
class TempFixConfig(FairseqDataclass):
    data_dir: str = field(
        default=os.getenv("TEMP_FIX_DATA_DIR"),
        metadata={"help": "path to image and temp directory"},
    )


@register_task("temp_fix", dataclass=TempFixConfig)
class TempFixTask(FairseqTask):

    def __init__(self, config: TempFixConfig):
        super().__init__(config)
        self.data_dir = Path(config.data_dir)
        data_list = os.listdir(self.data_dir/"images")
        self.data_list = [
            x.replace(".bmp", "") for x in data_list if x.endswith(".bmp")
        ]
        self.train_list, self.val_list = sklearn.model_selection.train_test_split(
            self.data_list, test_size=0.2
        )

    def load_dataset(self, split, **kwargs):

        if split == "train":
            self.datasets[split] = TempFixDataset(self.data_dir, self.train_list)
        elif split == "valid":
            self.datasets[split] = TempFixDataset(self.data_dir, self.val_list)
        else:
            raise KeyError(f"Invalid split: {split}")

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
