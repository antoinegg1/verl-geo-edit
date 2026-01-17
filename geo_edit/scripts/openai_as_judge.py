import os
import re
from dataclasses import dataclass
from typing import List, Optional, Union
from ..constants import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT 
from openai import OpenAI

ANSWER_TEMPLATE = "<answer>{}</answer>"

def parse_score(text: str) -> str:
    """
    从模型输出里抽取 Score: 0/1，返回 "0" 或 "1"；找不到返回空串。
    """
    m = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return m.group(1) if m else ""


def extract_answer(text: str, mode: str) -> Optional[str]:
    """
    从模型输出中抽取ANSWER_TEMPLATE，抽不到返回 None。
    mode:
      - "split": 宽松 split
      - "strict": 任意位置匹配 ANSWER_TEMPLATE
    """
    parts=ANSWER_TEMPLATE.split("{}")
    if mode == "split":
        if parts[0] not in text or parts[1] not in text:
            return None
        return text.split(parts[0])[-1].split(parts[1])[0].strip()
    
    if mode == "strict":
        start = text.find(parts[0])
        if start == -1:
            return None
        start += len(parts[0])
        if parts[1]:
            end = text.find(parts[1], start)
            if end == -1:
                return None
            return text[start:end].strip()
        return text[start:].strip()


    raise ValueError(f"Unknown extract mode: {mode}")


def get_final_prediction(predict_str_list: List[str], extract_mode: Optional[str]) -> str:
    """
    只取最后一轮作为最终输出；若 extract_mode 非空，则尝试抽取 <answer>...</answer>。
    抽取失败时，退回使用最后一轮原文（strip 后）。
    """
    if not predict_str_list:
        return ""
    last = predict_str_list[-1].strip()
    if not extract_mode:
        return last
    extracted = extract_answer(last, extract_mode)
    return extracted if extracted is not None else last

class OpenAIJudge:
    """
    使用 OpenAI 官方 SDK + Responses API 做 0/1 判分。
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def judge_correctness(self, question: str, ground_truth: str, prediction: str) -> str:
        prompt = EVAL_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        resp = self.client.responses.create(
            model=self.model,
            instructions=EVAL_SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        )
        return parse_score(resp.output_text or "")


def evaluate_final_answer(
    question: str,
    predict_str_list: List[str],
    ground_truth: str,
    cfg: Optional[dict] = None,
) -> Union[float, dict]:
    """
    只评价最终答案正确性：
      - 返回 1.0 / 0.0
      - 若没解析到 Score，则返回 dict 方便你上层过滤
    """

    final_pred = get_final_prediction(predict_str_list, cfg.extract_answer_tags)

    judge = OpenAIJudge(model=cfg.model)
    score_str = judge.judge_correctness(question, ground_truth, final_pred)

    if score_str == "":
        return {"is_filter": True, "info": "no_score_returned", "raw_final_pred": final_pred}
    return 1.0 if score_str == "1" else 0.0


if __name__ == "__main__":
    question = "Elena Ferrante"
    predict_str_list = [
        """<think>To determine the name ...</think> <grounding>{"bbox_2d": [2761, 715, 3160, 896]}</grounding>""",
        """<think>To determine the name ...</think> <answer>The name of the store with a blue sign is "J&optica."</answer>""",
    ]
    ground_truth = "Jptica"

    cfg = {
        "model": "gpt-4.1-mini",
        "extract_answer_tags": "strict",
    }

    result = evaluate_final_answer(question, predict_str_list, ground_truth, cfg)
    print(result)
