#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import jsonlines
from tqdm import tqdm
import re
import csv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import get_acc, keep_reference, get_rouge, get_bert_score, get_f1_score


def setup_pipeline(model_type, server_addr):
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig
    if model_type == 'InternLM2':
        if "7B" in server_addr:
            backend_config = TurbomindEngineConfig(model_name="internlm2-chat-7b", tp=1)
            pipe = pipeline(server_addr,
                            model_name="internlm2-chat-7b",
                            backend_config=backend_config,
                            chat_template_config=ChatTemplateConfig(model_name='internlm2-chat-7b'))
        elif "20B" in server_addr:
            backend_config = TurbomindEngineConfig(model_name="internlm2-chat-20b", tp=2)
            pipe = pipeline(server_addr,
                            model_name="internlm2-chat-20b",
                            backend_config=backend_config,
                            chat_template_config=ChatTemplateConfig(model_name='internlm2-chat-20b'))
        gen_config = GenerationConfig(stop_words=["", "[UNUSED_TOKEN_145]"])
        return pipe, gen_config
    elif model_type == 'Llama2':
        if "70b" in server_addr:
            tp = 4
        elif "13b" in server_addr:
            tp = 2
        else:
            tp = 1
        backend_config = TurbomindEngineConfig(model_name="llama-2-chat", tp=tp)
        pipe = pipeline(server_addr,
                        backend_config=backend_config,
                        model_name="llama-2-chat",
                        chat_template_config=ChatTemplateConfig(model_name='llama-2-chat'))
        return pipe, None
    elif model_type == 'Baichuan2':
        tp = 2 if "13B" in server_addr else 1
        backend_config = TurbomindEngineConfig(model_name='baichuan2-7b', tp=1)
        pipe = pipeline(server_addr,
                        model_name="baichuan2-7b",
                        backend_config=backend_config,
                        chat_template_config=ChatTemplateConfig(model_name='baichuan2-7b'))
        return pipe, None
    elif model_type == 'Qwen':
        if "7B" in server_addr:
            backend_config = TurbomindEngineConfig(model_name="qwen-7b", tp=1)
            pipe = pipeline(server_addr,
                            model_name="qwen-7b",
                            backend_config=backend_config,
                            chat_template_config=ChatTemplateConfig(model_name='qwen-7b'))
        elif "14B" in server_addr:
            backend_config = TurbomindEngineConfig(model_name="qwen-14b", tp=2)
            pipe = pipeline(server_addr,
                            backend_config=backend_config,
                            model_name="qwen-14b",
                            chat_template_config=ChatTemplateConfig(model_name='qwen-14b'))
        return pipe, None
    else:
        return None, None

def generate_response(model_type, pipe, gen_config, prompt):
    if model_type == 'InternLM2':
        reply = pipe(prompt, gen_config=gen_config).text
        res = re.search("<\|im_end\|>", reply)
        if res:
            reply = reply[:res.span()[0]]
    elif model_type == 'Llama2':
        reply = pipe(prompt).text
    elif model_type == 'Baichuan2':
        reply = pipe(prompt).text
    elif model_type == 'Qwen':
        reply = pipe(prompt).text
    else:
        reply = None
    return reply

def process_model_answer_input_sent(args, output_path, pipe, gen_config):
    lines = []

    with jsonlines.open(args.json_path) as f:
        lines = list(f)

    for pi, turns in tqdm(enumerate(lines), total=len(lines)):
        assert turns[0]["role"] == "user"
        prompt = turns[0]["content"]
        assert turns[1]["role"] == "assistant"
        golden = turns[1]["content"]

        reply = generate_response(args.model_type, pipe, gen_config, prompt)

        with jsonlines.open(output_path, "a") as writer:
            writer.write({"prompt": prompt, "reply":reply, "golden":golden})

def evaluate_output(output_path, eval_sorce_path):
    acc, zh_acc, en_acc = get_acc(output_path)
    pre_sore = keep_reference(output_path)
    zh_rouge_sore, en_rouge_sore, overall_rouge = get_rouge(output_path)
    zh_bert_score, en_bert_score, overall_bert_score = get_bert_score(output_path)
    f1_score = get_f1_score(output_path)
    
    if eval_sorce_path:
        with open(eval_sorce_path, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerow(["ACC", "zh_ACC", "en_ACC", 
                             "PRE", "F1",
                             "zh-ROUGEL", "en-ROUGEL", "overall-ROUGEL",
                             "zh-bert_score", "en-bert_score", "overall-bert_score",])
            writer.writerow([acc, zh_acc, en_acc,
                             pre_sore, f1_score,
                             zh_rouge_sore, en_rouge_sore, overall_rouge,
                             zh_bert_score, en_bert_score, overall_bert_score])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['InternLM2', 'Llama2', 'Baichuan2', 'Qwen'])
    parser.add_argument('--server_addr',  type=str)
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--eval_sorce_path", type=str)

    args = parser.parse_args()
    
    output_path = args.output_path
    model_type = args.model_type
    eval_sorce_path = args.eval_sorce_path

    pipe, gen_config = setup_pipeline(args.model_type, args.server_addr)

    process_model_answer_input_sent(args, output_path, pipe, gen_config)

    evaluate_output(output_path, eval_sorce_path)
