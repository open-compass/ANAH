import json, jsonlines
import re
import json
import jsonlines

def sentence_tokenize_process_dot(text, recover=False):
    if not recover:
        text = re.sub(r"O\.S\.B\.M. ", r"O.S.B.M.", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2\3\4", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Za-z])", r"\1\2\3", text)  # J. K. Campbell
        text = re.sub(r"((\n\s*)|(\. ))(\d+)\.\s+", r"\1\4.", text) #1. XXX
        text = re.sub(r"^(\d+)\.\s+", r"\1.", text) #1. XXX
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.\s+", r"\1\2.", text)
        text = re.sub(r"(\W|^)(et al)\.\s+([a-z])", r"\1\2.\3", text)
        text = re.sub(r"Alexander v\. Holmes", r"Alexander v.Holmes", text)
        text = re.sub(r"Brown v\. Board", r"Brown v.Board", text)
    else:
        text = re.sub(r"^(\d+)\.", r"\1. ", text) #1. XXX
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2 \3 \4", text) # J. K. Campbell
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z][a-z])", r"\1\2 \3", text)  # J. Campbell
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.", r"\1\2. ", text)
        text = re.sub(r"(\W|^)(et al)\.([a-z])", r"\1\2. \3", text)
        
        text = re.sub("O\.S\.B\.M\.", "O.S.B.M. ", text)
        text = re.sub("U\. +S\.", "U.S.", text)
        text = re.sub("U\.S\. *S\. *R\.", "U.S.S.R.", text)
        text = re.sub("D\. +C\.", "D.C.", text)
        text = re.sub("D\. +Roosevelt", "D. Roosevelt", text)
        text = re.sub("A\. *D\. *(\W)", r"A.D.\1", text)
        text = re.sub("A\. +D\.", r"A.D.", text)
        text = re.sub("F\. +C\.", r"F.C.", text)
        text = re.sub("J\. +League", r"J.League", text)
        text = re.sub(r"Alexander v\. *Holmes", r"Alexander v. Holmes", text)
        text = re.sub(r"Brown v\. *Board", r"Brown v. Board", text)
    return text

def sentence_tokenize(text, language, keep_end, keep_colon=False):
    if language == 'zh':
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1", text)
        sents2 = []
        sents = re.split("(。|！|？|；|\n+)", text) 
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sents2.append(sent)
        # print(sents2)
        return sents2  
    elif language == 'en':
        text = sentence_tokenize_process_dot(text)
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1 ", text)
        
        sents2 = []
        sents = re.split("((?:[.!?;]\s+)|(?:\n+))", text)
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sent = sentence_tokenize_process_dot(sent, recover=True)
                sents2.append(sent)
        return sents2 

def get_lines_from_path(input_path):
    if type(input_path) == str and input_path.endswith("jsonl"):
        with jsonlines.open(input_path) as f:
            lines = list(f)
    elif type(input_path) == str and input_path.endswith("json"):
        with open(input_path, 'r') as f:
            lines = json.load(f)
    elif type(input_path) == list:
        lines = input_path
    else:
        print(input_path)
        assert False
    return lines

def write_lines_to_path(output_path, lines):
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in lines:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    return f"Data saved successfully in {output_path}!"