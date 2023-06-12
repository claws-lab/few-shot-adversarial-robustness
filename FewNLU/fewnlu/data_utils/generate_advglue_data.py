import os
import json
import csv 

data_json = "./data/advGLUE/dev/dev.json"
output_dir = "./data/original/"


task_dir_map = {
    "sst2": "SST-2",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "rte": "RTE",
    "qnli": "QNLI",
    "qqp": "QQP",
}


def get_file_name(task, tag = "_adv", ext = ".tsv"):    
    
    task_dir = os.path.join(output_dir, task_dir_map[task])

    if task == "mnli":
        file_name = "dev_matched" + tag + ext
    elif task == "mnli-mm":
        file_name = "dev_mismatched" + tag + ext
    else:            
        file_name = "dev" + tag + ext
    
    output_file = os.path.join(task_dir, file_name)

    return output_file

class PreProcessor:
    def __init__(self):
        self.task = ""

    def get_reference_header(self):
        reference_file = get_file_name(self.task, tag = "")
        with open(reference_file,'rt') as f:
            tsv_reader = csv.reader(f, delimiter = "\t")            
            for line in tsv_reader:
                header = line
                break
            return header
    
    def get_header_map(self):
        return {}    
    
    def get_label_map(self):
        return {}

    def preprocess_data(self, dict_data):

        ref_header = self.get_reference_header()
        header_map = self.get_header_map()
        label_map = self.get_label_map()
        rows = []
        rows.append(ref_header)    
        for line in dict_data:
            row = []
            for column in ref_header:
                dict_key = header_map.get(column, column)
                # print(dict_key)
                value = line.get(dict_key, "")
                if dict_key == "label":
                    value = label_map.get(value, value)
                
                # print(value)
                row.append(value)
            # print(row)
            rows.append(row)

        return rows
    

class Sst2PreProcessor(PreProcessor):
    def __init__(self):
        super().__init__()
        self.task = "sst2"

    def get_label_map(self):
        return {0: "0", 1: "1"}

class MnliPreProcessor(PreProcessor):
    def __init__(self):
        super().__init__()
        self.task = "mnli"
    
    def get_header_map(self):
        return {"index": "idx", "sentence1": "premise", "sentence2": "hypothesis", "gold_label": "label"}

    def get_label_map(self):
        return { 0: "entailment", 1: "neutral", 2: "contradiction"}

class QqpPreProcessor(PreProcessor):
    def __init__(self):
        super().__init__()
        self.task = "qqp"
    
    def get_header_map(self):
        return {"id": "idx", "is_duplicate": "label"}
    
    def get_label_map(self): 
        return {0: "0", 1: "1"}

class RtePreProcessor(PreProcessor):
    def __init__(self):
        super().__init__()
        self.task = "rte"
    
    def get_header_map(self):
        return {"index": "idx"}
    
    def get_label_map(self): 
        return {0: "entailment", 1: "not_entailment"}

class QNLIPreProcessor(PreProcessor):
    def __init__(self):
        super().__init__()
        self.task = "qnli"
    
    def get_header_map(self):
        return {"index": "idx"}
    
    def get_label_map(self): 
        return {0: "entailment", 1: "not_entailment"}


pre_processor_map = {
    "sst2": Sst2PreProcessor,
    "mnli": MnliPreProcessor,
    "mnli-mm": MnliPreProcessor,
    "qqp": QqpPreProcessor,
    "rte": RtePreProcessor,
    "qnli": QNLIPreProcessor,
}

print("parsing the advGLUE file")
with open(data_json) as f:

    data_dict = json.load(f)

    for task in task_dir_map:
        
        print(task)
        if task not in data_dict:
            raise "task not found in advGLUE dev json file"

        output_file = get_file_name(task)

        with open(output_file, "wt", newline = "\n") as g:
            tsv_writer = csv.writer(g, delimiter = "\t", quoting=csv.QUOTE_NONE, quotechar='', escapechar='')
            
            pre_processor = pre_processor_map[task]()
            rows = pre_processor.preprocess_data(data_dict[task])

            tsv_writer.writerows(rows)


    



            

                

         



