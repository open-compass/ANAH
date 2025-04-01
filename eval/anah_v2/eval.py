import os
import argparse
import csv
from tqdm import tqdm

from utils import get_lines_from_path, write_lines_to_path, sentence_tokenize
from anahv2_prompt import fact_check_prompt, reference_check_prompt, hallucination_check_prompt
from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig

def process_question(pipe, question: str, sentence: str, document: str, language: str) -> str:
    """Process the question through multiple steps: fact-checking, reference-checking, hallucination-checking."""
    
    # Step 1: Fact-checking
    fact_check = fact_check_prompt(question, sentence, language)
    messages = [{"role": "user", "content": fact_check}]
    response = pipe(messages).text

    if response == "<No Facts>" or response == "<无事实>":
        return "nofact"

    messages.append({"role": "assistant", "content": response})

    # Step 2: Reference-checking
    reference_check = reference_check_prompt(question, document, sentence, language)
    messages.append({"role": "user", "content": reference_check})
    response = pipe(messages).text

    response_tmp = response.strip().replace(" ", "").lower()
    if "noreferenceinformation" in response_tmp or "无参考信息" in response_tmp:
        return "unverifiable"

    reference = response
    messages.append({"role": "assistant", "content": reference})

    # Step 3: Hallucination-checking
    hallucination_check = hallucination_check_prompt(question, reference, sentence, language)
    messages.append({"role": "user", "content": hallucination_check})
    response = pipe(messages).text

    hallucination_type = response.strip().replace(" ", "").lower()
    if "nohallucination" in hallucination_type or "无幻觉" in hallucination_type:
        return "ok"
    elif "contradictory" in hallucination_type or "矛盾" in hallucination_type:
        return "contradictory"
    elif "unverifiable" in hallucination_type or "无法验证" in hallucination_type:
        return "unverifiable"


def evaluate_output(args):
    lines = get_lines_from_path(args.annotation_path)
    
    total, contradictory, unverifiable, ok, nofact = 0, 0, 0, 0, 0
    for line in lines:
        total += 1
        annotation = line["annotation"]
        if annotation == "contradictory":
            contradictory += 1
        if annotation == "unverifiable":
            unverifiable += 1
        if annotation == "ok":
            ok += 1
        if annotation == "nofact":
            nofact += 1
    
    hallu = contradictory + unverifiable

    with open(args.eval_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ok", "nofact", "contradictory", "unverifiable", "hallucination", "total", "score"])
        writer.writerow([
            f"{(ok/total)*100:.2f}%", 
            f"{(nofact/total)*100:.2f}%", 
            f"{(contradictory/total)*100:.2f}%", 
            f"{(unverifiable/total)*100:.2f}%", 
            hallu, 
            total, 
            f"{(1 - hallu/total)*100:.2f}", 
        ])


def run(args):
    path = 'opencompass/anah-v2'
    backend_config = TurbomindEngineConfig(model_name="internlm2", tp=1)
    chat_template_config = ChatTemplateConfig(model_name='internlm2')
    pipe = pipeline(path, backend_config=backend_config, chat_template_config=chat_template_config)

    old_lines = get_lines_from_path("eval/anah_v2/question_document.jsonl")
    ref_lines = dict()
    for line in old_lines:
        ref_lines[line["question"]] = dict(document=line["document"], language=line["language"])
    
    lines = get_lines_from_path(args.json_path)
    new_lines = []
    for line in tqdm(lines, desc="processing"):
        response = line["predictions"]
        question = line["prompt"][-1]["content"]
        language = ref_lines[question]["language"]
        document = ref_lines[question]["document"]
        sentences = sentence_tokenize(response, language, keep_end=False)
        for sent in sentences:
            annotation = process_question(pipe, question, sent, document, language)
            new_lines.append({"question": question, "response": response, "sentence": sent, "annotation": annotation, "language": language})

    write_lines_to_path(args.annotation_path, new_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--document_path", default="eval/anah_v2/question_document.jsonl", type=str)
    parser.add_argument("--annotation_path", type=str)
    parser.add_argument("--eval_path", type=str)
    args = parser.parse_args()

    if os.path.exists(args.eval_path):
        print("Already evaluated")
        exit(0)

    if not os.path.exists(args.annotation_path):
        run(args)

    evaluate_output(args)

