import json
import os
import re
import numpy as np
import json_repair
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  
import argparse

# --- Configuration ---
JUDGE_PROMPT = """You are an impartial and rigorous evaluator. Your task is to determine if the [Model Prediction] is correct based on the provided [Question] and [Golden Label].

### Evaluation Criteria:
1. **Accuracy**: The prediction must match the key information in the Golden Label.  If the decimal places of the prediction answer and the golden answer are different, please use the rounding method.
2. **Semantic Equivalence**: Differences in phrasing, synonyms, or sentence structure are acceptable as long as the meaning remains strictly the same.
3. **Redundancy**: 
    - If the prediction contains the correct answer plus extra helpful/neutral information, mark it as Correct (1).
    - If the prediction contains the correct answer but adds contradictory or false information, mark it as Incorrect (0).
4. **Format**: Ignore differences in capitalization, punctuation, or minor spacing.

### Input Data:
- **Question**: {question}
- **Golden Answer**: {golden_label}
- **Model Prediction**: {pred_answer}

### Output Format:
Please output a strictly valid JSON object. Do not include any Markdown formatting (like ```json). The JSON must contain:
- "reasoning": (string) A brief explanation comparing the prediction to the label.
- "result": (integer) 1 for Correct, 0 for Incorrect.

### Example Output:
```json
{{"reasoning": "The prediction accurately captures the core concept of the golden label, despite using different wording.", "result": 1}}
```

Please output the JSON object directly, do not include any other text. You only need to judge whether the model prediction is equal to the golden lable.
"""

RESEARCH_JUDGE_PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction. Your task is to comprehensively evaluate the quality of the generated data science report, based on the provided user instruction [INSTRUCTION], a checklist offering reference points for an ideal report [CHECKLIST], and the generated report [REPORT].

Evaluate across two dimensions (1–5 scale):

- **Content**: Relevance, comprehensiveness, and insightfulness.
- **Format**: Structure, readability, and professionalism.

### [INSTRUCTION]:
{question}

### [CHECKLIST]:
{checklist}

### [REPORT]:
{pred_answer}

Return your evaluation strictly as JSON:
```json
{{
"Content": <score>,
"Format": <score>
}}
```"""

argparser = argparse.ArgumentParser()
argparser.add_argument('--output_dir', type=str, required=True,
                       help='Directory holding the per-item JSON predictions to score.')
argparser.add_argument('--judge_model', type=str,
                       default=os.environ.get('JUDGE_MODEL', 'Qwen/QwQ-32B'),
                       help='Judge model id passed to the OpenAI-compatible endpoint.')
argparser.add_argument('--judge_api_url', type=str,
                       default=os.environ.get('JUDGE_API_URL', 'http://localhost:8000/v1'))
argparser.add_argument('--judge_api_key', type=str,
                       default=os.environ.get('JUDGE_API_KEY', 'EMPTY'))
argparser.add_argument('--max_workers', type=int,
                       default=int(os.environ.get('JUDGE_MAX_WORKERS', '30')))
args = argparser.parse_args()

output_dir = args.output_dir
dataset_name = output_dir.rstrip('/').split('/')[-1].split('-')[0]
model_name = args.judge_model
api_url = args.judge_api_url
api_key = args.judge_api_key

client = OpenAI(api_key=api_key, base_url=api_url)
MAX_WORKERS = args.max_workers

def process_single_item(item_name):
    """Score a single prediction file with the judge model."""
    file_path = os.path.join(output_dir, item_name)
    
    
    with open(file_path, 'r', encoding='utf-8') as f:
        item = json.load(f)
    
    question = item['question']
    golden_answer = item['question_item'].get('answer', '')

    if 'research' in dataset_name:
        prediction = item['prediction']
    else:
        if 'short_answer' in item:
            prediction = item['short_answer']
        elif 'prediction' in item:
            prediction = item['prediction']
        else:
            if not item.get('running_messages'):
                print("no valid running messages")
                prediction = "No valid prediction"
            else:
                prediction = item['running_messages'][-1]['content']

    pred_answer = prediction

    # Choose evaluation mode based on dataset
    if 'research' in dataset_name:
        if pred_answer is None:
            metrics = {'content_score': 0, 'format_score': 0, 'reasoning': 'no valid answer found'}
        else:
            checklist = item['question_item']['checklist']
            input_prompt = RESEARCH_JUDGE_PROMPT.format(question=question, checklist=checklist, pred_answer=pred_answer)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": input_prompt}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=8192,
            )
            
            result_content = response.choices[0].message.content
            result_content = result_content.split("</think>")[-1].strip()
            match = re.search(r"```(?:json)?(.*?)```", result_content, re.DOTALL)
            score_str = match.group(1).strip() if match else result_content
            result_json = json_repair.loads(score_str)
            content_score = result_json.get('Content', 0)
            format_score = result_json.get('Format', 0)
            metrics = {'content_score': content_score, 'format_score': format_score, 'reasoning': result_content}
    
    else: 
        if pred_answer is None:
            metrics = {'llm_judge': 0, 'em': 0, 'reasoning': 'no valid answer found'}
        else:
            em = int(str(pred_answer).strip() == str(golden_answer).strip())
            input_prompt = JUDGE_PROMPT.format(question=question, golden_label=golden_answer, pred_answer=pred_answer)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": input_prompt}],
                temperature=0.0,
                max_tokens=4000,
            )
            
            result_content = response.choices[0].message.content
            result_content = result_content.split("</think>")[-1].strip()
            match = re.search(r"```(?:json)?(.*?)```", result_content, re.DOTALL)
            score_str = match.group(1).strip() if match else result_content
            result_json = json_repair.loads(score_str)
            if not isinstance(result_json, dict):
                print("judge result is not a valid json!")
                llm_judge = 0
                print(result_content)
                print("----------")
            else:
                llm_judge = int(result_json.get('result', 0))
        
            metrics = {
                'llm_judge': llm_judge,
                'em': em,
                'reasoning': result_content
            }
    metrics['detect_answer'] = pred_answer
    item['judge_result'] = metrics
    
    return item


def main():
    all_files = os.listdir(output_dir)
    target_files = [f for f in all_files if f.endswith(".json") and not f.startswith("_") and 'graph' not in f]

    result_list = []
    
    print(f"Judging {len(target_files)} files with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_item, f): f for f in target_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(target_files), desc="Judging"):
            result = future.result()
            if result is not None:
                result_list.append(result)
    # Sort by id for stable output
    result_list = sorted(result_list, key=lambda x: int(x['idx']))

    print("Evaluation done. Building summary...")
    type2score = {}
    for item in result_list:
        if dataset_name == 'tablebench':
            qtypes = item['question_item'].get('qtype', 'unknown')
        elif 'infiagent' in dataset_name:
            qtypes = item.get('concepts', 'unknown')
        elif 'research' in dataset_name:
            qtypes = item['question_item']['type']
        else:
            qtypes = ['unknown']
            
        if isinstance(qtypes, str):
            qtypes = [qtypes]
        
        keys = [key for key in item['judge_result'].keys() if key not in ['reasoning', 'detect_answer']]
        for qtype in qtypes:
            if qtype not in type2score:
                type2score[qtype] = {key: [] for key in keys}
                type2score[qtype]['num_samples'] = 0
                type2score[qtype]['valid_samples'] = 0
            for key in keys:
                type2score[qtype][key].append(item['judge_result'][key])
            detect_answer = item['judge_result']['detect_answer']
            type2score[qtype]['num_samples'] += 1
            if detect_answer is not None:
                type2score[qtype]['valid_samples'] += 1

    summary = {}
    for qtype, score in type2score.items():
        summary[qtype] = {}
        for k,v in score.items():
            if k == 'num_samples':
                continue
            summary[qtype][k] = round(np.mean(v), 4)
        summary[qtype]['num_samples'] = score['num_samples']
        summary[qtype]['valid_samples'] = score['valid_samples']
    summary['global'] = {}
    for k in keys:
        summary['global'][k] = round(np.mean([item['judge_result'][k] for item in result_list]), 4)
    summary['global']['num_samples'] = len(result_list)
    

    with open(os.path.join(output_dir, '_final_judge_results.json'), 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
        
    with open(os.path.join(output_dir, '_final_judge_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()