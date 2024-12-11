from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
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

# Initialize the pipeline outside the function
path = 'opencompass/anah-v2'
backend_config = TurbomindEngineConfig(model_name="internlm2", tp=1)
chat_template_config = ChatTemplateConfig(model_name='internlm2')
pipe = pipeline(path, backend_config=backend_config, chat_template_config=chat_template_config)

# Example of calling the function
question = ""
sentence = ""
document = ""
language = ""

label = process_question(pipe, question, sentence, document, language)
print(label)
