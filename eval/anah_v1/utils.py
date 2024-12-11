from evaluate import load
import re
from rouge_chinese import Rouge
import jieba 
import csv
import jsonlines
import argparse
import math
import numpy as np
from bert_score import score as bert_score
import collections
from sklearn.metrics import f1_score


def get_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]


def get_ngram_counter(tokens, n):
    ngrams = get_ngrams(tokens, n)
    counter = collections.Counter()
    counter.update(ngrams)
    return counter


def _prec_recall_f1_score(pred_items, gold_items, language, n=1):
    assert language in ["en", "zh"]
    if language == "en":
        pred_items = pred_items.split()
        gold_items = gold_items.split()
        
    pred_items = get_ngram_counter(pred_items, n)
    gold_items = get_ngram_counter(gold_items, n)
    common = gold_items & pred_items
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def get_sent_refs(ann_sent, keep_format):
    lines = ann_sent.split('\n')
    for ref_i, line in enumerate(lines):
        if re.search("<(参考|Reference)>", line, flags=re.I):
            break
    if not re.search("(参考)|(Reference)", lines[ref_i], flags=re.I):
        return []
        
    sent_refs = []
    for i in range(ref_i, len(lines)):
        line = lines[i]
        if re.search("<(改正|Correction)>", line, flags=re.I):
            break
        if not keep_format:
            line = re.sub("<(参考|Reference)>", "", line, flags=re.I)
            line = re.sub("((参考)|(Reference))\d+：", "", line, flags=re.I).strip()
        for ref in line.split("<SEP>"):
            ref = ref.strip()
            ref = re.sub("""参考文档中的""", "", ref)
            ref = re.sub("""^["”“'’‘]|["”“'’‘]$""", "", ref).strip()
            ref = re.sub("""[,.，。;；]$""", "", ref).strip()
            if ref and ref.lower() not in ["无", "none"]:
                sent_refs.append(ref)
    return sent_refs


def get_lines_from_path(input_path):
    if type(input_path) == str and input_path.endswith("jsonl"):
        with jsonlines.open(input_path) as f:
            lines = list(f)
    elif type(input_path) == list:
        lines = input_path
    else:
        assert False
    return lines


def get_type(text):
    if re.search("(<无事实>)|(<No ?Fact>)", text, re.IGNORECASE):
        return "nofact"
    for line in text.split("\n"):
        if re.search("(<幻觉>无法验证)|(<Hallucination> ?Unverifiable)", line, re.IGNORECASE):
            return "unverifiable"
        if re.search("(<幻觉>无)|(<Hallucination> ?None)", line, re.IGNORECASE):
            return "ok"
        if re.search("(<幻觉>矛盾)|(<Hallucination> ?Contradictory)", line, re.IGNORECASE):
            return "contradictory"
    return -1


def detect_language(prompt):
    if re.search("(参考文档)|(标注)", prompt):
        language = 'zh'
    elif re.search("(annotat)|(Reference)", prompt):
        language = 'en'
    else:
        assert False
    return language


def keep_reference(input_path):
    lines = get_lines_from_path(input_path)
    pre_sore3, pre_sore4 = [], []
    zh_pre_sore4, en_pre_sore4 = [], []

    for i, line in enumerate(lines):
        ann_sent = line["reply"]
        golden = line["golden"]
        prompt = line['prompt']
        language = detect_language(prompt)
        ref = get_sent_refs(ann_sent, keep_format=False)
        if ref:
            ref = " ".join(ref).lower()
            if language == 'zh':
                doc_s = re.search("\n参考((文档)|(文件)|(文章))：", prompt).span()[1]
                doc_e = re.search("\n((((请标记)|(请分析)|(开始评估)|(请标注))((答案)|(回答)|(要点)|(信息点))?))：", prompt).span()[0]
            else:
                try:
                    doc_s = re.search("\nReference: ", prompt).span()[1]
                except:
                    print(prompt)
                    
                doc_e = re.search("\n(Please )?((annotate)|(mark)|(analyze)) (the )?((answer)|((information )?point)): ", prompt, re.IGNORECASE).span()[0]
                
            doc = prompt[doc_s:doc_e].lower()
            precision3, _, _ = _prec_recall_f1_score(ref, doc, language, n=3)
            pre_sore3.append(precision3)
            precision4, _, _ = _prec_recall_f1_score(ref, doc, language, n=4)
            pre_sore4.append(precision4)
            if language == 'en':
                en_pre_sore4.append(precision4)
            else:
                zh_pre_sore4.append(precision4)

    pre_sore3 = round(np.mean(pre_sore3) * 100, 2)
    pre_sore4 = round(np.mean(pre_sore4) * 100, 2)
    
    en_pre_sore4 = round(np.mean(en_pre_sore4) * 100, 2)
    zh_pre_sore4 = round(np.mean(zh_pre_sore4) * 100, 2)
    
    print(f"precision3/4: {pre_sore3}/{pre_sore4}")
    print(f"zh precision4: {zh_pre_sore4} en precision4: {en_pre_sore4}")
    return f"{pre_sore3}/{pre_sore4}"


def get_acc(input_path):
    lines = get_lines_from_path(input_path)
    print("acc len(lines)", len(lines))
    acc_list = []
    zh_acc_list, en_acc_list = [], []
    for i, line in enumerate(lines):
        prompt = line["prompt"]
        language = detect_language(prompt)
        
        golden_type = get_type(line['golden'])
        if golden_type == -1:
            print(line['golden'])
        assert golden_type != -1

        pred_type = get_type(line['reply'])
        if golden_type == pred_type:
            acc_list.append(1)
            if language == 'zh':
                zh_acc_list.append(1)
            else:
                en_acc_list.append(1)
        else:
            acc_list.append(0)
            if language == 'zh':
                zh_acc_list.append(0)
            else:
                en_acc_list.append(0)
                
    acc = round(np.mean(acc_list)*100, 2) 
    zh_acc = round(np.mean(zh_acc_list)*100, 2) 
    en_acc = round(np.mean(en_acc_list)*100, 2) 
    print(f"ACC: {acc}%")
    print(f"zh ACC: {zh_acc}% en ACC {en_acc}%")
    return acc, zh_acc, en_acc


# https://github.com/baiyyang/BLEU    
def calculate_ngram(candidates, references, n, language):
    count_clip = 0
    count = 0
    for index, candidate in enumerate(candidates):
        references_list = lines2dic(references, index, n, language)
        if language == "en":
            words = candidate.split()
        else:
            words = candidate
        limit = len(words) - n + 1
        candidate_dic = {}
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in candidate_dic.keys():
                candidate_dic[key] += 1
            else:
                candidate_dic[key] = 1
        count_clip += clip(candidate_dic, references_list)
        count += limit
    if count_clip == 0:
        pr = 0
    else:
        pr = float(count_clip) / count
    return pr


def brevity_penalty(candidates, references, language):
    c = 0
    r = 0
    for index, candidate in enumerate(candidates):
        c_length = len(candidate.split()) if language == "en" else len(candidate)
        reference_index = [reference[index] for reference in references]
        r_lengths = [len(r.split()) if language == "en" else len(r) for r in reference_index]
        c += c_length
        r += match_reference(c_length, r_lengths)
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - float(r) / c)
    return bp


def match_reference(candidate_len, reference_lens):
    best_len = abs(reference_lens[0] - candidate_len)
    best_ref = reference_lens[0]
    for length in reference_lens:
        if abs(length - candidate_len) < best_len:
            best_len = abs(length - candidate_len)
            best_ref = length
    return best_ref


def clip(candidate, references):
    count = 0
    for cand in candidate.keys():
        cand_value = candidate[cand]
        max_count = 0
        for reference in references:
            if cand in reference.keys():
                max_count = max(reference[cand], max_count)
        count += min(max_count, cand_value)
    return count


def lines2dic(references, index, n, language):
    reference_list = []
    for reference in references:
        reference_dic = {}
        line = reference[index]
        if language == "en":
            words = line.split()
        else:
            words = line
        limit = len(words) - n + 1
        for i in range(limit):
            key = " ".join(words[i: i+n]).lower() if language == "en" else words[i: i+n]
            if key in reference_dic.keys():
                reference_dic[key] += 1
            else:
                reference_dic[key] = 1
        reference_list.append(reference_dic)
    return reference_list


def geometric_mean(precisions):
    return math.exp(sum([math.log(p) if p != 0 else -math.inf for p in precisions]) / len(precisions))


def read_files_rouge(generated_path, target_path=''):
    if type(generated_path) == list or \
        (type(generated_path) == str and generated_path.endswith("jsonl")):
        lines = get_lines_from_path(generated_path)
        zh_generated_txt, zh_target_txt, en_generated_txt, en_target_txt = [], [], [], []
        for i, line in enumerate(lines):
            prompt = line['prompt']
            golden = line["golden"]
            reply = line["reply"]
            language = detect_language(prompt)
            golden = re.sub("<\|.*?\|>:?", "", golden)
            golden = re.sub("<TOKENS_UNUSED_\d+>", "", golden)
            reply = re.sub("<TOKENS_UNUSED_\d+>", "", reply)
            
            if language == 'en':
                en_target_txt.append(golden.lower().strip())
                en_generated_txt.append(reply.lower().strip())
            else:
                zh_target_txt.append(' '.join(jieba.cut(golden)))
                zh_generated_txt.append(' '.join(jieba.cut(reply)))

    elif generated_path.endswith("txt"):
        with open(target_path) as f:
            target_txt = []
            for l in f.readlines():
                if language == 'en':
                    target_txt.append(l.lower().strip())
                else:
                    target_txt.append(' '.join(jieba.cut(l)))

        with open(generated_path) as f:
            generated_txt = []
            for l in f.readlines():
                if language == 'en':
                    generated_txt.append(l.lower().strip())
                else:
                    generated_txt.append(' '.join(jieba.cut(l)))
                    
    elif generated_path.endswith("csv"):
        target_txt, generated_txt = [], []
        with open(generated_path) as f:
            lines = list(csv.reader(f))
            head = lines[0]
            for i, key in enumerate(head):
                if key == 'golden':
                    golden_i = i
                elif key == 'reply':
                    reply_i = i
                    
            for line in lines[1:]:
                l = line[golden_i]
                l = re.sub("<\|.*?\|>:?", "", l)
                l = re.sub("<TOKENS_UNUSED_\d+>", "", l)
                if language == 'en':
                    target_txt.append(l.lower().strip())
                    generated_txt.append(line[reply_i].lower().strip())
                else:
                    target_txt.append(' '.join(jieba.cut(l)))
                    l = line[reply_i]
                    generated_txt.append(' '.join(jieba.cut(l)))
                    
    assert len(zh_generated_txt)==len(zh_target_txt)
    assert len(en_generated_txt)==len(en_target_txt)
    return zh_generated_txt, zh_target_txt, en_generated_txt, en_target_txt


def get_rouge(input_path):
    zh_rouge_sore, en_rouge_sore = -1, -1
    zh_generated_txt, zh_target_txt, en_generated_txt, en_target_txt = read_files_rouge(input_path)
    zh_len = len(zh_generated_txt)
    en_len = len(en_generated_txt)
    
    if zh_len:
        rouge = Rouge()
        rouge_sore = []
        for g, t in zip(zh_generated_txt, zh_target_txt):
            if g:
                score = rouge.get_scores(g, t, avg=True)
                rouge_sore.append(score["rouge-l"]['f'])
            else:
                rouge_sore.append(0)
        zh_rouge_sore = round(np.mean(rouge_sore) * 100, 2)
        print("zh RougeL", zh_rouge_sore)
    if en_len:
        rouge = load('rouge')
        results = rouge.compute(predictions=en_generated_txt, references=en_target_txt)
        en_rouge_sore = round(results["rougeL"]*100, 2)
        print("en RougeL", en_rouge_sore)
        
    overall_rouge = (zh_len*zh_rouge_sore + en_len*en_rouge_sore)/ (zh_len+en_len)
    print(F"overall ROUGE {overall_rouge}")   
        
    return zh_rouge_sore, en_rouge_sore, overall_rouge
        
        
def get_bert_score(input_path):
    zh_generated_txt, zh_target_txt, en_generated_txt, en_target_txt = read_files_rouge(input_path)
    zh_len = len(zh_generated_txt)
    en_len = len(en_generated_txt)
    
    if zh_len:
        _, _, zh_F1 = bert_score(zh_generated_txt, zh_target_txt, model_type="bert-base-chinese", lang="zh", verbose=True)
        zh_F1 = round(zh_F1.mean().item()*100, 2)
        print("zh bert_score", zh_F1)
    if en_len:
        _, _, en_F1 = bert_score(en_generated_txt, en_target_txt, lang="en", verbose=True)
        en_F1 = round(en_F1.mean().item()*100, 2)
        print("en bert_score", en_F1)
        
    overall_bert_score = (zh_len*zh_F1 + en_len*en_F1)/ (zh_len+en_len)
    print(F"overall bert_score {overall_bert_score}")   
    return zh_F1, en_F1, overall_bert_score


def get_type_f1(text):
    if re.search("(<无事实>)|(<No ?Fact>)", text, re.IGNORECASE):
        return 0
    for line in text.split("\n"):
        if re.search("(<幻觉>无法验证)|(<Hallucination> ?Unverifiable)", line, re.IGNORECASE):
            return 1
        if re.search("(<幻觉>无)|(<Hallucination> ?None)", line, re.IGNORECASE):
            return 2
        if re.search("(<幻觉>矛盾)|(<Hallucination> ?Contradictory)", line, re.IGNORECASE):
            return 3
    return -1


def get_f1_score(input_path):
    lines = get_lines_from_path(input_path)
    gold_list = []
    pre_list = []
    for i, line in enumerate(lines):
        prompt = line["prompt"]
        
        golden_type = get_type_f1(line['golden'])
        gold_list.append(golden_type)
        if golden_type == -1:
            print(line['golden'])
        assert golden_type != -1

        pred_type = get_type_f1(line['reply'])
        pre_list.append(pred_type)

    f1 = f1_score(gold_list, pre_list, average='weighted')
    print("f1 score ", f1)
    return f1
