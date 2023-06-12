import os

from typing import List
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor
from utils import InputExample
# from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics

class GLUEDataProcessor(DataProcessor):
    def __init__(self, task_name):
        super().__init__()
        assert task_name in GLUE_PROCESSORS

#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

#     def get_test_examples(self, data_dir, tag = None):
#         """See base class."""
#         set_type = "test" + tag if tag else "test"
#         print(set_type)
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, set_type + ".tsv")), set_type)



# class Sst2Processor(GLUEDataProcessor):
#     """Processor for the SST-2 data set (GLUE version)."""

#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence"].numpy().decode("utf-8"),
#             None,
#             str(tensor_dict["label"].numpy()),
#         )

#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training, dev and test sets."""
#         examples = []
#         text_index = 0
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = line[text_index]
#             label = line[1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#         return examples


# class MnliProcessor(DataProcessor):
#     """Processor for the MultiNLI data set (GLUE version)."""

#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["premise"].numpy().decode("utf-8"),
#             tensor_dict["hypothesis"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )

#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

#     def get_test_examples(self, data_dir, tag = None):
#         """See base class."""
#         set_type = "test_matched"+ tag if tag else "test_matched"
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, set_type+".tsv")), set_type)

#     def get_labels(self):
#         """See base class."""
#         return ["contradiction", "entailment", "neutral"]

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training, dev and test sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[8]
#             text_b = line[9]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


# class MnliMismatchedProcessor(MnliProcessor):
#     """Processor for the MultiNLI Mismatched data set (GLUE version)."""

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

#     def get_test_examples(self, data_dir, tag = None):
#         """See base class."""
#         set_type = "test_mismatched"+ tag if tag else "test_mismatched"
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, set_type + ".tsv")), set_type)

class RteProcessor(GLUEDataProcessor):
#     """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, use_cloze):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, tag=None, adv = False):
        """See base class."""
        filename = "test_adv.tsv" if adv else "test.tsv"
        tag = "test_adv" if adv else "test"        
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx = i))
        return examples

class MnliProcessor(GLUEDataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, use_cloze):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir, adv = False) -> List[InputExample]:
        filename = "test_matched_adv.tsv" if adv else "test_matched.tsv"
        tag = "test_matched_adv" if adv else "test_matched"

        print("\n\nI am here!!! \n\n")
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "unlabeled_matched.tsv")), "unlabeled_matched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx = i)
            examples.append(example)

        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI mismatched data set (GLUE version)."""

    # def get_dev_examples(self, data_dir):
    #     return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir, adv = False) -> List[InputExample]:
        filename = "test_mismatched_adv.tsv" if adv else "test_mismatched.tsv"
        tag = "test_mismatched_adv" if adv else "test_mismatched"
        print("\n\n hurrayyyy!!! \n\n")
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "unlabeled_mismatched.tsv")), "unlabeled_matched")


class Sst2Processor(GLUEDataProcessor):
    """Processor for the SST2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, use_cloze):
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, adv = False) -> List[InputExample]:
        filename = "test_adv.tsv" if adv else "test.tsv"
        tag = "test_adv" if adv else "test"        
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(Sst2Processor._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, idx = i))
        return examples

class QnliProcessor(GLUEDataProcessor):
#     """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, use_cloze):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, tag=None, adv = False):
        """See base class."""
        filename = "test_adv.tsv" if adv else "test.tsv"
        tag = "test_adv" if adv else "test"        
        return self._create_examples(QnliProcessor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(QnliProcessor._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx = i))
        return examples

class QqpProcessor(GLUEDataProcessor):
#     """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, use_cloze):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, tag=None, adv = False):
        """See base class."""
        filename = "test_adv.tsv" if adv else "test.tsv"
        tag = "test_adv" if adv else "test"        
        return self._create_examples(QqpProcessor._read_tsv(os.path.join(data_dir, filename)), tag)

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(QqpProcessor._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""

        q1_index = 3 if set_type != "unlabeled" else 1
        q2_index = 4 if set_type != "unlabeled" else 2
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[-1]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx = i))
        return examples


GLUE_PROCESSORS = {    
    "rte": RteProcessor,
    "sst2": Sst2Processor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,    
} 


