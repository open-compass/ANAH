import os
import argparse
import csv
import json
import time
from tqdm import tqdm
import torch
from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig

from utils import get_lines_from_path, write_lines_to_path, sentence_tokenize
from anahv2_prompt import fact_check_prompt, reference_check_prompt, hallucination_check_prompt


def process_batch(pipe, batch_data, retry_on_error=True):
    """Process a batch of questions with retry mechanism."""
    results = []
    for item in tqdm(batch_data):
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
                    # Small delay before retry
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
        elif annotation == "error":
            error += 1
    
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


def create_pipeline(model_path, gpus):
    """Create a pipeline with the specified GPU configuration."""
    backend_config = TurbomindEngineConfig(
        model_name="internlm2", 
        tp=len(gpus.split(',')) if gpus else 1
    )
    
    # Set visible devices before initializing the pipeline
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        
    chat_template_config = ChatTemplateConfig(model_name='internlm2')
    
    try:
        return pipeline(model_path, backend_config=backend_config, chat_template_config=chat_template_config)
    except Exception as e:
        print(f"Error creating pipeline: {str(e)}")
        raise


def load_checkpoint(checkpoint_path):
    """Load checkpoint file to resume processing."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"processed_count": 0}


def save_checkpoint(checkpoint_path, processed_count):
    """Save checkpoint for resuming later."""
    with open(checkpoint_path, 'w') as f:
        json.dump({"processed_count": processed_count}, f)


def run(args):
    model_path = 'anah-v2'
    checkpoint_path = f"{args.annotation_path}.checkpoint"
    
    # Load checkpoint if it exists
    checkpoint = load_checkpoint(checkpoint_path)
    processed_count = checkpoint["processed_count"]
    
    # Load reference data
    old_lines = get_lines_from_path(args.document_path)
    ref_lines = dict()
    for line in old_lines:
        ref_lines[line["question"]] = dict(document=line["document"], language=line["language"])
    
    # Load input data
    lines = get_lines_from_path(args.json_path)
    
    # Check if annotation file exists and load it
    new_lines = []
    if os.path.exists(args.annotation_path) and processed_count > 0:
        new_lines = get_lines_from_path(args.annotation_path)
        print(f"Resuming from checkpoint with {processed_count} items already processed.")
    
    # Prepare data for processing
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
    
    # Skip already processed items
    remaining_items = all_batch_items[processed_count:]
    print(f"Total items to process: {len(remaining_items)}")
    
    if not remaining_items:
        print("All items already processed.")
        write_lines_to_path(args.annotation_path, new_lines)
        return
    
    # Initialize the pipeline
    pipe = None
    
    # Process in batches
    batch_size = args.batch
    for i in range(0, len(remaining_items), batch_size):
        batch = remaining_items[i:i+batch_size]
        
        # Create or recreate pipeline if needed
        if pipe is None:
            try:
                pipe = create_pipeline(model_path, args.gpus)
            except Exception as e:
                print(f"Failed to create pipeline: {str(e)}")
                # Wait a bit before retrying
                time.sleep(5)
                try:
                    pipe = create_pipeline(model_path, args.gpus)
                except Exception as e2:
                    print(f"Failed to create pipeline again: {str(e2)}")
                    break
        
        # Process the batch
        try:
            results = process_batch(pipe, batch)
            
            # Add results to output
            for j, result in enumerate(results):
                item = batch[j]
                new_lines.append({
                    "question": item["question"],
                    "response": item["response"],
                    "sentence": item["sentence"],
                    "annotation": result,
                    "language": item["language"]
                })
            
            # Update processed count and save checkpoint
            processed_count += len(batch)
            save_checkpoint(checkpoint_path, processed_count)
            
            # Periodically save results
            if i % (batch_size * 5) == 0 or i + batch_size >= len(remaining_items):
                write_lines_to_path(args.annotation_path, new_lines)
                print(f"Saved progress: {processed_count}/{len(all_batch_items)} items processed")
                
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory. Recreating the pipeline...")
            # Force garbage collection
            if pipe:
                del pipe
            torch.cuda.empty_cache()
            pipe = None
            time.sleep(5)  # Wait a bit before retrying
            
            # Retry the current batch with a new pipeline
            try:
                pipe = create_pipeline(model_path, args.gpus)
                results = process_batch(pipe, batch)
                
                for j, result in enumerate(results):
                    item = batch[j]
                    new_lines.append({
                        "question": item["question"],
                        "response": item["response"],
                        "sentence": item["sentence"],
                        "annotation": result,
                        "language": item["language"]
                    })
                
                processed_count += len(batch)
                save_checkpoint(checkpoint_path, processed_count)
                write_lines_to_path(args.annotation_path, new_lines)
                
            except Exception as e:
                print(f"Failed to recover from OOM: {str(e)}")
                # Save what we have so far
                write_lines_to_path(args.annotation_path, new_lines)
                break
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Try to recover by recreating the pipeline
            if pipe:
                del pipe
            torch.cuda.empty_cache()
            pipe = None
            time.sleep(5)
            
            # Save progress before potentially crashing
            write_lines_to_path(args.annotation_path, new_lines)
    
    # Final save
    write_lines_to_path(args.annotation_path, new_lines)
    print(f"Processing complete. Total items processed: {processed_count}/{len(all_batch_items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, help="Path to input JSON file")
    parser.add_argument("--document_path", default="question_document.jsonl", type=str, 
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
