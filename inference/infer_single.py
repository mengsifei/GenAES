from utils.data_utils import *
from utils.prompts import *
import torch
from my_app import * 

def replace_single_newlines(text):
    return re.sub(r'(?<!\n)\n(?!\n)', '\\\\n\\\\n', text)

def generate_full_prompt(topic, essay, cefr_stat):
    essay = replace_single_newlines(essay)
    paragraph_cnt = len(essay.replace('\\n\\n', '\\n').split('\\n'))
    word_cnt = len(essay.split())
    stat = f"The essay has {word_cnt} words and {paragraph_cnt} paragraphs.\n"
    full_prompt = feedback_prompt_1 + "\n ##{{PROMPT}}\n```\n" + topic + "\n```\n ##{{ESSAY}}\n```\n" + essay + "\n```\n##CEFR Analysis\n" + str(cefr_stat) + "\n\n##{{STATS}}\n" + stat + '\n' + feedback_prompt_2
    return full_prompt


def generate_and_score_essay(topic, essay):
    global MODELS_LOADED, LONGFORMER_TOKENIZER, LONGFORMER_MODEL, QWEN_TOKENIZER, QWEN_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LONGFORMER_MODEL = LONGFORMER_MODEL.to(device)
    QWEN_MODEL = QWEN_MODEL.to(device)

    cefr_results = get_cefr_stats(essay)
    full_prompt = generate_full_prompt(topic=topic, essay=essay, cefr_stat=cefr_results)
    essay = replace_single_newlines(essay)
    paragraph_cnt = len(essay.replace('\\n\\n', '\\n').split('\\n'))
    text = QWEN_TOKENIZER.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
    inputs = QWEN_TOKENIZER(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side='left'
    ).to(device)
    with torch.inference_mode():
        outputs = QWEN_MODEL.generate(
            **inputs,
            max_new_tokens=1500,
            use_cache=True, 
            pad_token_id=QWEN_TOKENIZER.eos_token_id
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    full_feedback = QWEN_TOKENIZER.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # Fix spaces/newlines
    )
    output_match = re.search(r"{(.*?)}", full_feedback, re.DOTALL)
    response = output_match.group(1).strip() if output_match else full_feedback
    feedback_components = extract_feedback_keys_values(response)
    feedback_components = dict(feedback_components)
    feedback_components['word_count'] = len(essay.split())
    feedback_components['paragraph_count'] = paragraph_cnt
    feedback_components['cefr_stat'] = cefr_results
    score_input = create_train_input({
    'topic': topic,
    'essay': essay,
    'Corrected_essay': feedback_components.get('Corrected_essay', ''),
    'TR_feedback': feedback_components.get('TR_feedback', ''),
    'CC_feedback': feedback_components.get('CC_feedback', ''),
    'LR_feedback': feedback_components.get('LR_feedback', ''),
    'GRA_feedback': feedback_components.get('GRA_feedback', ''),
    'word_count':feedback_components.get('word_count', ''),
    'paragraph_count': feedback_components.get('paragraph_count', ''),
    'cefr_stat': feedback_components.get('cefr_stat', '')
    })
    score_inputs = LONGFORMER_TOKENIZER(
        score_input,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True
    ).to(device)
    LONGFORMER_MODEL.eval()
    with torch.no_grad():
        outputs = LONGFORMER_MODEL(**score_inputs)  # Get full outputs dictionary
        scores = outputs['logits'].cpu().numpy()
    scores = [round(x) for x in scores[0]]
    return scores, feedback_components