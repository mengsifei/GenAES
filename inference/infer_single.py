from transformers import LongformerTokenizer, AutoTokenizer, AutoModelForCausalLM
from models.model import CustomLongformerForSequenceClassification
from utils.data_utils import *
from utils.prompts import *
import torch
import time
total_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/Longformer_checkpoint/checkpoint-21690"
model = CustomLongformerForSequenceClassification.from_pretrained(
    model_path,
    problem_type="regression",
)
model = model.to(device)
model.eval()
long_former_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model_name = 'Qwen/Qwen3-1.7B'


qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, device='auto')
qwen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).half().to(device)

qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id


def generate_and_score_essay(row):
    topic = row['Topic']
    essay = row['Essay']
    cefr_results = get_cefr_stats(essay)
    full_prompt = row['full_prompt']
    text = qwen_tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
    inputs = qwen_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side='left'
    ).to(device)
    start = time.time()
    with torch.inference_mode():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=1500,
            use_cache=True, 
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    # Alternative decoding (ensure no special tokens leak)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    full_feedback = qwen_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # Fix spaces/newlines
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    # print(full_feedback)
    output_match = re.search(r"{(.*?)}", full_feedback, re.DOTALL)
    response = output_match.group(1).strip() if output_match else full_feedback
    # print(response)

    feedback_components = extract_feedback_keys_values(response)
    
    # print(feedback_components)
    # print(cefr_results)
    score_input = create_train_input({
    'topic': topic,
    'essay': essay,
    'Corrected_essay': feedback_components.get('Corrected_essay', ''),
    'TR_feedback': feedback_components.get('TR_feedback', ''),
    'CC_feedback': feedback_components.get('CC_feedback', ''),
    'LR_feedback': feedback_components.get('LR_feedback', ''),
    'GRA_feedback': feedback_components.get('GRA_feedback', ''),
    'word_count': len(essay.split()),  # Auto-calculate if not provided
    'paragraph_count': len(essay.split('\n\n')),  # Auto-calculate if not provided
    'cefr_stat': cefr_results
    })
    print(score_input)
    start = time.time()
    score_inputs = long_former_tokenizer(
        score_input,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**score_inputs)  # Get full outputs dictionary
        scores = outputs['logits'].cpu().numpy()
        # print(scores)
    end = time.time()
    print(f"Time taken LONGFORMER: {end - start} seconds")
    return scores

topic = 'some people say that the best way to improve public health is by increasing the number of sports facilities. others, however, say that this would have little effect on public health and that other measures are required. discuss both these views and give your own opinion'
essay = "There is an ongoing debate over how to develop public health. Many individuals claim that increasing the number of sports amenities might improve the population's health, while I believe that it is a less effective method to develop healthcare and other solutions needed. \\n\\nOn the one hand, sport is crucial to keep healthy and it improves physical activity, which leads to a better body. According to  doctors and sportsmen, individuals, who are engaged in the sport as a part of their daily lives, tend to have  better longevity. If individuals are involved in sports, this tendency has an inevitable influence on public health. Additionally, the main way to prevent cardiovascular diseases is by doing sports. For example, people have a preference to go for a sport in Finland, which contributes to a decline in  heart disease by  improving  public health.\\n\\nOn the other hand, a lack of awareness of people and a  shortage of healthcare professionals is the most important in order to develop the country's healthcare system. In other words, authorities should divert the  budget from sports facilities to  certain problems. If  society does not have any information about the cure or prevent themselves from the ill, it does not matter if they engage in  sports. That is why, authorities have to improve  public awareness about the prevention of the virus. Moreover, a lack of  literate doctors is an obstacle to improving the healthcare system. For example, there is a vast number of  doctors in Turkey; as a result, it affects on the better society's health.\\n\\nIn conclusion, although doing sports ensures  our muscular system in order to keep healthier, I am of the opinion that other measures, including, increasing  public awareness about  certain sicknesses and the number of doctors, are  more significant in order to improve  public health."
essay = replace_single_newlines(essay)
cefr = "{'A1_words': 152, 'A2_words': 65, 'B1_words': 27, 'B2_words': 14, 'C1_words': 0, 'C2_words': 11, 'unknown_words': 43, 'total_words': 312, 'A1_pct': 48.717948717948715, 'A2_pct': 20.833333333333336, 'B1_pct': 8.653846153846153, 'B2_pct': 4.487179487179487, 'C1_pct': 0.0, 'C2_pct': 3.5256410256410255, 'unknown_pct': 13.782051282051283}"
stat = "The essay has 269 words and 4 paragraphs."
tmp = {"Topic": topic, "Essay": essay, "full_prompt": feedback_prompt_1 + "\n ##{{PROMPT}}\n```\n" + topic + "\n```\n ##{{ESSAY}}\n```\n" + essay + "\n```\n##CEFR Analysis\n" + cefr + "\n\n##{{STATS}}n" + stat + '\n' + feedback_prompt_2}
# print(tmp['full_prompt'])
print(generate_and_score_essay(tmp))
print("Total time: ", time.time() - total_time)