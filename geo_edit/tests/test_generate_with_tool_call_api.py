import argparse
import json
import logging
import os
import shutil
from ..agents.api_agent import APIBasedAgent, AgentConfig
from ..environment.action import TOOL_FUNCTIONS
from ..environment.task.vision_qa_task import VisionQATask
from ..config import NOTOOL_INPUT_TEMPLATE, build_agent_configs
from ..constants import SYSTEM_PROMPT, MAX_TOOL_CALLS
from ..utils.logger import setup_logger
logger = setup_logger(__name__)

def main():
    # argparse 
    parser = argparse.ArgumentParser(description="Generate content with tool calls using Google GenAI API.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for Google GenAI.")
    parser.add_argument("--dataset_path", type=str, required=False, help="Unused for the test script.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-pro-preview", help="Model name or path.")
    parser.add_argument("--max_concurrent_requests", type=int, default=32, help="Maximum number of concurrent requests.")
    args = parser.parse_args()
    
    api_key= args.api_key
    output_path= args.output_dir
    os.makedirs(output_path, exist_ok=True)
    max_output_tokens= None

    agent_configs = build_agent_configs(
        max_output_tokens=max_output_tokens,
        thinking_level="high",
        include_thoughts=True,
        temperature=1.0,
        system_prompt=SYSTEM_PROMPT,
        candidate_count=1,
        tool_mode="AUTO",
        disable_automatic_function_calling=True,
    )
    direct_generate_config = agent_configs.direct_generate_config

    config = AgentConfig(
        model_type="Google",
        model_name=args.model_name_or_path,
        api_key=api_key,
        generate_config=direct_generate_config,
        n_retry=3,
    )
    api_agent=APIBasedAgent(config)

    meta_info_list= []

    INPUT_TEMPLATE= NOTOOL_INPUT_TEMPLATE
    test_id = "test_case"
    if os.path.exists(os.path.join(output_path, test_id)):
        with open(os.path.join(output_path, test_id, "meta_info.jsonl"), "r", encoding="utf-8") as f:
            meta_info= json.loads(f.readline().strip())
            meta_info_list.append(meta_info)
        logging.info(f"Example id: {test_id} already processed, skipping.")
    else:
        api_agent.reset()
        task_save_dir= os.path.join(output_path, test_id)
        os.makedirs(task_save_dir, exist_ok=True)
        question= "User input: In the picture on the right a number should be written next to each point. The sum of the numbers on the corners of each side of the hexagon should be equal. Two numbers have already been inserted. Which number should be in the place marked '$x$'?\n<image1>"
        options= ""
        answer= ""
        image_url = "234.jpg"

        text_prompt= INPUT_TEMPLATE.format(question=question, options=options)

        task= VisionQATask(
            task_id=test_id,
            task_prompt=text_prompt,
            task_answer=answer,
            task_image_path=image_url,
            tool_functions=TOOL_FUNCTIONS,
            save_dir=task_save_dir,
            options=options,
        )
        max_tool_calls = MAX_TOOL_CALLS
        for i in range(max_tool_calls):
            try:
                action, extra_info = api_agent.act(task.contents)
                function_call_part_list = task.parse_action(step=i+1, action=action, extra_info=extra_info)
            except Exception as e:
                task.state = False
                shutil.rmtree(task_save_dir)
                logging.error(f"Error during agent action for example id: {test_id} at step {i+1}: {e}")
                break

            if not function_call_part_list or not function_call_part_list[-1].function_call:
                logging.info("Final response generated without further tool calls.")
                break

            task.update_observation_from_action(function_call_part_list)
        else:
            logging.info("Max tool calls reached; forcing final answer without tool calls.")
            FORCE_ANSWER_PROMPT = "Max tool calls reached. Please provide the final answer without further tool calls."
            task.contents.append(FORCE_ANSWER_PROMPT)
            api_agent.config.generate_config = agent_configs.force_generate_config
            action, extra_info = api_agent.act(task.contents)

            _ = task.parse_action(step=max_tool_calls + 1, action=action, extra_info=extra_info)

        if task.state:
            meta_info = task.save_trajectory()
            meta_info_list.append(meta_info)

    total_tool_calls = 0
    total_tokens = 0
    tool_usage_counts = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0
    for info in meta_info_list:
        total_tool_calls += info["function_call_total_count"]
        total_tokens += info["tokens_used_total"]
        if info["total_steps"] >= MAX_TOOL_CALLS :
            reach_max_tool_call_count += 1
        if info["function_call_total_count"] == 0:
            direct_answer_count += 1
        for tool_name, count in info["function_call_each_count"].items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + count

    global_meta_info = {
        "total_examples": len(meta_info_list),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
    }
    global_meta_info_jsonl_path = os.path.join(output_path, "global_meta_info.jsonl")
    with open(global_meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_meta_info) + "\n")
            
if __name__ == "__main__":
    main()
