import fire
import re
from tqdm import tqdm
import time
import json
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os


def evaluate(jsonl_file, save_path, regen_answer=False):
    lines = load_jsonl(jsonl_file)
    for line in tqdm(lines, desc='gen_correct'):
        gt_answer = line['answer'].strip()
        if len(line['options']) > 0:
            gt_answer_value = line['options'][ord(gt_answer)-ord('A')]
        else:
            gt_answer_value = ''

        if 'model_answer' not in line or regen_answer:
            model_answer = line['generated_answer'].strip()
            for c in 'ABCDE':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                
            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    save_jsonl(save_path, lines, t_stamp=False)


def math_level_subject_acc(jsonl_file):

    lines = load_jsonl(jsonl_file)
    
    results_dict = {}
    for line in tqdm(lines, desc='math_level_subject_acc'):
        correct = line['correct']
        subject = line['subject']
        level = line['level']
        for key in [
            '-all', 
            f'-level{level}', 
            f'{subject}', 
            f'{subject}_level{level}'
            ]:
            if key not in results_dict:
                results_dict[key] = [0,0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1


    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(jsonl_file), ':\t', results_dict['-all'])
    json.dump(results_dict, open(jsonl_file.replace('.jsonl', '_result.json'), 'w'), indent=4, ensure_ascii=False)



if __name__ == '__main__':
    jsonl_path = 'MathVision_gemini_clean.jsonl'
    save_path = 'MathVision_gemini_clean_evaluated.jsonl'
    evaluate(jsonl_path, save_path, True)
    math_level_subject_acc(save_path)