import os
import argparse
import csv
import json
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig

from utils import get_lines_from_path, write_lines_to_path, sentence_tokenize
from anahv2_prompt import fact_check_prompt, reference_check_prompt, hallucination_check_prompt


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
    
    return "unverifiable"  # Default fallback


def process_batch(pipe, batch_data, retry_on_error=True):
    results = []
    for item in tqdm(batch_data, desc="Processing batch", leave=False):
        try:
            result = process_question(
                pipe, 
                item["question"], 
                item["sentence"], 
                item["document"], 
                item["language"]
            )
            results.append(result)
        except Exception as e:
            if retry_on_error:
                print(f"Error processing question: {str(e)}. Retrying...")
                try:
                    time.sleep(1)
                    result = process_question(
                        pipe, 
                        item["question"], 
                        item["sentence"], 
                        item["document"], 
                        item["language"]
                    )
                    results.append(result)
                except Exception as e2:
                    print(f"Retry failed: {str(e2)}. Skipping this item.")
                    results.append("unverifiable")
            else:
                print(f"Error processing question: {str(e)}. Skipping.")
                results.append("unverifiable")
    
    return results


def load_local_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
    return {"processed_items": [], "last_index": 0}


def save_local_checkpoint(checkpoint_path, processed_items, last_index):
    data = {
        "processed_items": processed_items,
        "last_index": last_index
    }

    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(data, f)
    os.replace(temp_path, checkpoint_path)


def run_distributed(rank, world_size, args, remaining_items, processed_count, gpu_ids):
    if rank < len(gpu_ids):
        gpu_id = gpu_ids[rank]

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Process {rank} using GPU {gpu_id}")
    else:
        print(f"Process {rank} has no GPU assigned, using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    print(f"Process {rank}: Initialized distributed process group")
    
    items_per_process = len(remaining_items) // world_size
    start_idx = rank * items_per_process
    end_idx = start_idx + items_per_process if rank < world_size - 1 else len(remaining_items)
    
    local_items = remaining_items[start_idx:end_idx]
    print(f"Process {rank}: Assigned {len(local_items)} items from {start_idx} to {end_idx-1}")

    checkpoint_path = f"{args.annotation_path}.checkpoint.rank{rank}"
    checkpoint = load_local_checkpoint(checkpoint_path)
    last_processed_index = checkpoint["last_index"]
    processed_items = checkpoint["processed_items"]
    
    if last_processed_index > 0:
        local_items = local_items[last_processed_index:]
        print(f"Process {rank}: Resuming from index {last_processed_index}, {len(processed_items)} items already processed")
    
    model_path = 'opencompass/anah-v2'
    try:
        backend_config = TurbomindEngineConfig(model_name="internlm2", tp=1)
        chat_template_config = ChatTemplateConfig(model_name='internlm2')
        pipe = pipeline(model_path, backend_config=backend_config, chat_template_config=chat_template_config)
        print(f"Process {rank}: Created pipeline successfully")
    except Exception as e:
        print(f"Process {rank}: Failed to create pipeline: {str(e)}")
        dist.destroy_process_group()
        return None

    batch_size = args.batch
    
    try:
        for i in range(0, len(local_items), batch_size):
            batch = local_items[i:i+batch_size]
            current_global_indices = [start_idx + last_processed_index + i + j for j in range(len(batch))]
            
            print(f"Process {rank}: Processing batch {i//batch_size + 1}/{(len(local_items) + batch_size - 1)//batch_size}")
            
            try:
                batch_results = process_batch(pipe, batch)

                for j, result in enumerate(batch_results):
                    item = batch[j]
                    global_index = current_global_indices[j]
                    processed_items.append({
                        "question": item["question"],
                        "response": item["response"],
                        "sentence": item["sentence"],
                        "annotation": result,
                        "language": item["language"],
                        "global_index": global_index
                    })
                
                last_processed_index += len(batch)
                if i % (batch_size * 2) == 0 or i + batch_size >= len(local_items):
                    save_local_checkpoint(checkpoint_path, processed_items, last_processed_index)
                    print(f"Process {rank}: Saved checkpoint, processed {last_processed_index}/{len(local_items) + checkpoint['last_index']} items")
            
            except torch.cuda.OutOfMemoryError:
                print(f"Process {rank}: CUDA out of memory. Recreating the pipeline...")
                if pipe:
                    del pipe
                torch.cuda.empty_cache()
                time.sleep(5)
                
                try:
                    pipe = pipeline(model_path, backend_config=backend_config, chat_template_config=chat_template_config)
                    print(f"Process {rank}: Recreated pipeline after OOM")

                    continue
                    
                except Exception as e:
                    print(f"Process {rank}: Failed to recover from OOM: {str(e)}")
                    save_local_checkpoint(checkpoint_path, processed_items, last_processed_index)
                    break
            
            except Exception as e:
                print(f"Process {rank}: Error processing batch: {str(e)}")
                save_local_checkpoint(checkpoint_path, processed_items, last_processed_index)

                if pipe:
                    del pipe
                torch.cuda.empty_cache()
                time.sleep(5)
                
                try:
                    pipe = pipeline(model_path, backend_config=backend_config, chat_template_config=chat_template_config)
                    print(f"Process {rank}: Recreated pipeline after error")
                    continue
                except Exception as e:
                    print(f"Process {rank}: Failed to recreate pipeline: {str(e)}")
                    break
    
    except KeyboardInterrupt:
        print(f"Process {rank}: Interrupted, saving progress...")
        save_local_checkpoint(checkpoint_path, processed_items, last_processed_index)
    
    save_local_checkpoint(checkpoint_path, processed_items, last_processed_index)

    dist.barrier()
    
    print(f"Process {rank}: Completed processing {len(processed_items)} items")
    dist.destroy_process_group()
    
    return processed_items


def combine_results(world_size, args, new_lines, processed_count):
    print("Combining results from all processes...")
    
    all_results = []
    missing_rank = False
    
    # 从所有进程加载结果
    for rank in range(world_size):
        checkpoint_path = f"{args.annotation_path}.checkpoint.rank{rank}"
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    if "processed_items" in data:
                        all_results.extend(data["processed_items"])
                        print(f"Loaded {len(data['processed_items'])} results from process {rank}")
            except Exception as e:
                print(f"Error loading results from process {rank}: {str(e)}")
                missing_rank = True
        else:
            print(f"Warning: Missing checkpoint file for process {rank}")
            missing_rank = True
    
    if missing_rank:
        print("Warning: Some process results may be missing. Combined results may be incomplete.")
    
    # 按全局索引排序
    all_results.sort(key=lambda x: x.get("global_index", 0))
    
    # 移除辅助全局索引
    for item in all_results:
        if "global_index" in item:
            del item["global_index"]
    
    new_lines.extend(all_results)

    new_processed_count = processed_count + len(all_results)

    with open(f"{args.annotation_path}.checkpoint", 'w') as f:
        json.dump({
            "processed_count": new_processed_count,
            "last_combined_time": time.time(),
            "world_size": world_size
        }, f)

    write_lines_to_path(args.annotation_path, new_lines)
    print(f"Combined results saved. Total processed: {new_processed_count}")
    
    return new_processed_count


def evaluate_output(args):
    lines = get_lines_from_path(args.annotation_path)
    
    total, contradictory, unverifiable, ok, nofact, error = 0, 0, 0, 0, 0, 0
    for line in lines:
        total += 1
        annotation = line["annotation"]
        if annotation == "contradictory":
            contradictory += 1
        elif annotation == "unverifiable":
            unverifiable += 1
        elif annotation == "ok":
            ok += 1
        elif annotation == "nofact":
            nofact += 1
    
    hallu = contradictory + unverifiable

    with open(args.eval_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ok", "nofact", "contradictory", "unverifiable", "hallucination", "error", "total", "score"])
        writer.writerow([
            f"{(ok/total)*100:.2f}%", 
            f"{(nofact/total)*100:.2f}%", 
            f"{(contradictory/total)*100:.2f}%", 
            f"{(unverifiable/total)*100:.2f}%", 
            hallu, 
            error,
            total, 
            f"{(1 - hallu/total)*100:.2f}", 
        ])


def load_checkpoint(checkpoint_path):
    """Load checkpoint file to resume processing."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading main checkpoint: {str(e)}. Starting from beginning.")
    return {"processed_count": 0}


def run(args):
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires GPU support.")
        return
    
    checkpoint_path = f"{args.annotation_path}.checkpoint"
    
    checkpoint = load_checkpoint(checkpoint_path)
    processed_count = checkpoint["processed_count"]

    old_lines = get_lines_from_path(args.document_path)
    ref_lines = dict()
    for line in old_lines:
        ref_lines[line["question"]] = dict(document=line["document"], language=line["language"])
    
    lines = get_lines_from_path(args.json_path)
    
    new_lines = []
    if os.path.exists(args.annotation_path) and processed_count > 0:
        new_lines = get_lines_from_path(args.annotation_path)
        print(f"Resuming from checkpoint with {processed_count} items already processed.")
    
    all_batch_items = []
    
    for line in lines:
        response = line["response"]
        question = line["question"]
        if question not in ref_lines:
            print(f"Warning: Question '{question}' not found in reference data. Skipping.")
            continue
            
        language = ref_lines[question]["language"]
        document = ref_lines[question]["document"]
        sentences = sentence_tokenize(response, language, keep_end=False)
        
        for sent in sentences:
            all_batch_items.append({
                "question": question,
                "response": response,
                "sentence": sent,
                "document": document,
                "language": language
            })
    
    remaining_items = all_batch_items[processed_count:]
    print(f"Total items to process: {len(remaining_items)}")
    
    if not remaining_items:
        print("All items already processed.")
        write_lines_to_path(args.annotation_path, new_lines)
        return
    
    gpu_ids = []
    if args.gpus:
        gpu_ids = args.gpus.split(',')
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    
    world_size = len(gpu_ids)
    if world_size == 0:
        print("No GPUs available or specified. Exiting.")
        return
    
    print(f"Using {world_size} GPUs for distributed processing: {gpu_ids}")
    
    try:
        mp.spawn(
            run_distributed,
            args=(world_size, args, remaining_items, processed_count, gpu_ids),
            nprocs=world_size,
            join=True
        )

        new_processed_count = combine_results(world_size, args, new_lines, processed_count)
        
        print(f"Processing complete. Total items processed: {new_processed_count}/{len(all_batch_items)}")
    
    except KeyboardInterrupt:
        print("Main process interrupted. Will try to combine available results.")
        try:
            combine_results(world_size, args, new_lines, processed_count)
        except Exception as e:
            print(f"Error combining results after interrupt: {str(e)}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, help="Path to input JSON file")
    parser.add_argument("--document_path", default="eval/anah_v2/question_document.jsonl", type=str, 
                        help="Path to document JSON file")
    parser.add_argument("--annotation_path", type=str, help="Path to save annotation results")
    parser.add_argument("--eval_path", type=str, help="Path to save evaluation results")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated list of GPUs to use (e.g., '0,1')")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for processing")
    args = parser.parse_args()

    if os.path.exists(args.eval_path):
        print("Already evaluated")
        exit(0)

    if not os.path.exists(os.path.dirname(args.annotation_path)):
        os.makedirs(os.path.dirname(args.annotation_path), exist_ok=True)
        
    if not os.path.exists(os.path.dirname(args.eval_path)):
        os.makedirs(os.path.dirname(args.eval_path), exist_ok=True)

    run(args)
    evaluate_output(args)
