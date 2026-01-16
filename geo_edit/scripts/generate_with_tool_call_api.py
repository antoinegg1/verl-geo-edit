from google import genai
import io
import os
from google.genai import types
from PIL import Image
import requests
import argparse
from ..agents.api_agent import APIBasedAgent, AgentConfig
from ..environment.action import TOOL_FUNCTIONS, TOOL_FUNCTIONS_DECLARE
from ..environment.task.vision_qa_task import VisionQATask
from ..config import API_KEY
from ..constants import SYSTEM_PROMPT,MATHVISION_INPUT_TEMPLATE

import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # argparse 
    parser = argparse.ArgumentParser(description="Generate content with tool calls using Google GenAI API.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for Google GenAI.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-flash-preview", help="Model name or path.")
    parser.add_argument("--max_concurrent_requests", type=int, default=32, help="Maximum number of concurrent requests.")
    args = parser.parse_args()
    
    api_key= args.api_key
    dataset_path= args.dataset_path
    
    output_path= args.output_dir
    
    max_output_tokens=65536
    
    tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=list(TOOL_FUNCTIONS.keys())
        )
    )
    # tool_config = types.ToolConfig(
    #     function_calling_config=types.FunctionCallingConfig(
    #         mode="AUTO"
    #     )
    # )


    generate_config = types.GenerateContentConfig(
        tools=[tools],
        thinking_config=types.ThinkingConfig(
            thinkingLevel="low",
            include_thoughts=True
        ),
        tool_config=tool_config,
        temperature=1.0,
        system_instruction=[SYSTEM_PROMPT],
        max_output_tokens=max_output_tokens,
        candidate_count=1,
    )

    config = AgentConfig(
        model_type="Google",
        model_name=args.model_name_or_path,
        api_key=api_key,
        generate_config=generate_config,
        n_retry=3,
    )
    
    api_agent=APIBasedAgent(config)
    
    # Load dataset
    INPUT_TEMPLATE= MATHVISION_INPUT_TEMPLATE
    question= "User input: In the picture on the right a number should be written next to each point. The sum of the numbers on the corners of each side of the hexagon should be equal. Two numbers have already been inserted. Which number should be in the place marked '$x$'?\n<image1>"
    options=""
    image_url = "234.jpg"
    text_prompt= INPUT_TEMPLATE.format(question=question, options=options)
    
    task= VisionQATask(
        task_id="sample_task",
        task_prompt=text_prompt,
        task_answer="",
        task_image_path=image_url,
        tool_functions=TOOL_FUNCTIONS,
        save_dir=output_path,
    )
    
    max_tool_calls = 8
    for i in range(max_tool_calls):
        action, extra_info = api_agent.act(task.contents)
        
        function_call_part_list = task.parse_action(step=i+1, action=action, extra_info=extra_info)
    
        if not function_call_part_list or not function_call_part_list[-1].function_call:
            logging.info("Final response generated without further tool calls.")
            break
        
        task.update_observation_from_action(function_call_part_list)   
    else:
        logging.info("Max tool calls reached; forcing final answer without tool calls.")
        FORCE_ANSWER_PROMPT = "Max tool calls reached. Please provide the final answer without further tool calls."
        task.contents.append(FORCE_ANSWER_PROMPT)
        force_tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )
        force_generate_config = types.GenerateContentConfig(
            tools=[tools],
            thinking_config=generate_config.thinking_config,
            tool_config=force_tool_config,
            temperature=generate_config.temperature,
            system_instruction=generate_config.system_instruction,
            max_output_tokens=generate_config.max_output_tokens,
            candidate_count=generate_config.candidate_count,
        )
        api_agent.config.generate_config = force_generate_config
        action, extra_info = api_agent.act(task.contents)
        
        _ = task.parse_action(step=max_tool_calls + 1, action=action, extra_info=extra_info)
  
    task.save_trajectory()
            
if __name__ == "__main__":
    main()
