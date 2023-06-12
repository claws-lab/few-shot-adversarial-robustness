# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the different strategies to patternize data for all SuperGLUE tasks, including
direct concatenation, pattern-verbalizer pairs (PVPs), and ptuning PVPs.
"""


import string
from typing import List, Tuple, Union

from utils import InputExample, get_verbalization_ids
import log
from tasks.base_pvp import PVP, PVPOutputPattern

logger = log.get_logger()

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]

# class RtePVP(PVP):

#     VERBALIZER_A = {
#         "contradiction": ["Wrong"],
#         "entailment": ["Right"],
#         "neutral": ["Maybe"]
#     }
#     VERBALIZER_B = {
#         "contradiction": ["No"],
#         "entailment": ["Yes"],
#         "neutral": ["Maybe"]
#     }
#     def available_patterns(self):
#         if not self.use_cloze:
#             return [0]
#         elif not self.use_continuous_prompt:
#             return [0, 1, 2, 3]
#         else:
#             return [1, 2, 3, 4]

#     def get_parts(self, example: InputExample) -> FilledPattern:
#         text_a = self.shortenable(self.remove_final_punc(example.text_a))
#         text_b = self.shortenable(example.text_b)

#         if self.pattern_id == 0 or self.pattern_id == 2:
#             return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
#         elif self.pattern_id == 1 or self.pattern_id == 3:
#             return [text_a, '?'], [self.mask, ',', text_b]

#     def verbalize(self, label) -> List[str]:
#         if self.pattern_id == 0 or self.pattern_id == 1:
#             return RtePVP.VERBALIZER_A[label]
#         return RtePVP.VERBALIZER_B[label]

class RtePVP(PVP):

    _is_multi_token = False

    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    MULTI_VERBALIZER={
        "not_entailment": ["No","false"],
        "entailment": ["Yes","true"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        else:
            return [1, 2, 3, 4, 5, 6, 8, 10]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)  # premise
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis

        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            return [text_a], [text_b]

        elif not self.use_continuous_prompt:
            if self.pattern_id == 0 or self.pattern_id == 10:
                return ['"', text_b, '" ?'], [[self.mask_id], ', "', text_a, '"']
            elif self.pattern_id == 1 or self.pattern_id == 11:
                return [text_b, '?'], [[self.mask_id], ',', text_a]
            elif self.pattern_id == 2 or self.pattern_id == 12:
                return ['"', text_b, '" ?'], [[self.mask_id], '. "', text_a, '"']
            elif self.pattern_id == 3 or self.pattern_id == 13:
                return [text_b, '?'], [[self.mask_id], '.', text_a]
            elif self.pattern_id == 4 or self.pattern_id == 14:
                return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', [self.mask_id]], []
            elif self.pattern_id == 5 or self.pattern_id == 15:
                return [text_a, 'Question:', text_b, "?", "Answer:", [self.mask_id], "."], []

        else:
            if self.pattern_id == 1:
                return [text_a, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 2:
                return [text_a, 1, 'Question:', text_b, "?", 1, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 3:
                return [text_a, 1, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 4:
                return [text_a, 2, 'Question:', text_b, "?", 2, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 5:
                return [text_a, 2, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 6:
                return [text_a, 3, 'Question:', text_b, "?", 3, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 8:
                return [text_a, 4, 'Question:', text_b, "?", 4, "Answer:", [self.mask_id], "."], []

            elif self.pattern_id == 10:
                return [text_a, 5, 'Question:', text_b, "?", 5, "Answer:", [self.mask_id], "."], []


    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []
        
        return RtePVP.VERBALIZER[label]


# class QnliPVP(PVP):

#     VERBALIZER = {
#         "not_entailment": ["No"],
#         "entailment": ["Yes"]
#     }

#     def available_patterns(self):
#         if not self.use_cloze:
#             return [0]
#         elif not self.use_continuous_prompt:
#             return [0]
#         else:
#             return [0]

#     def get_parts(self, example: InputExample) -> PVPOutputPattern:
#         text_a = self.shortenable(example.text_a)  # premise
#         text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis

#         assert self.pattern_id in self.available_patterns()
        
#         return [text_a, '?'], [self.mask, ',', text_b]
#     def verbalize(self, label) -> List[str]:
        
#         return QnliPVP.VERBALIZER[label]

class QqpPVP(PVP):

    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3]
        else:
            return [0, 1, 2, 3]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)  # premise
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis

        assert self.pattern_id in self.available_patterns()        

        if self.pattern_id == 0:
            return [text_a], [self.mask, ',', text_b]
        elif self.pattern_id == 1:
            return [text_a], [self.mask, ', I want to know', text_b]
        elif self.pattern_id == 2:
            return [text_a], [self.mask, ', but', text_b]
        elif self.pattern_id == 3:
            return [text_a], [self.mask, ', please ,', text_b]
        else:
            raise ValueError("Invalid pattern id found")

        
    def verbalize(self, label) -> List[str]:
        
        return QqpPVP.VERBALIZER[label]

class QnliPVP(PVP):
    VERBALIZER_A = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]   
    }
    VERBALIZER_B = {
        "not_entailment": ["Wrong"],
        "entailment": ["Right"]   
    }
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3]
        else:
            return [1, 2, 3, 4]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return QnliPVP.VERBALIZER_A[label]
        return QnliPVP.VERBALIZER_B[label]


class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_B = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3]
        else:
            return [1, 2, 3, 4]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]

class Sst2PVP(PVP):
    VERBALIZER = {
        "0": ["bad"],
        "1": ["good"],
    }
    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [0, 1, 2, 3]
        else:
            return [0, 1, 2, 3]

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(self.remove_final_punc(example.text_a))

        # if self.pattern_id == 0:
        #     return [text_a], ["It was", self.mask, '.']

        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['In summary, the movie is', self.mask, '.']
        
        else:
            raise ValueError("Invalid pattern id found")

    def verbalize(self, label) -> List[str]:
        return Sst2PVP.VERBALIZER[label]


GLUE_PVPS = {
    'sst2': Sst2PVP,
    'mnli': MnliPVP,
    'mnli-mm': MnliPVP,
    'rte': RtePVP,
    'qnli': QnliPVP,
    'qqp': QqpPVP
}

GLUE_METRICS = {
    "sst2":  ["acc"],
    "mnli":  ["acc"],
    "mnli-mm":  ["acc"],
    "rte": ["acc"],
    "qnli": ["acc"],
    "qqp": ["acc", "f1"]
}