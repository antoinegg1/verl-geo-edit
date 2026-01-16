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
from datasets import load_dataset
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
    
    output_jsonl_path= args.output_dir
    os.makedirs(output_jsonl_path, exist_ok=True)
    output_jsonl_path= os.path.join(output_jsonl_path, "output.jsonl")
    extra_info_jsonl_path= os.path.join(args.output_dir, "extra_info.jsonl")
    meta_info_jsonl_path= os.path.join(args.output_dir, "meta_info.jsonl")
    image_save_dir= os.path.join(args.output_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    max_output_tokens=65536
    
    tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
    # tool_config = types.ToolConfig(
    #     function_calling_config=types.FunctionCallingConfig(
    #         mode="ANY", allowed_function_names=list(TOOL_FUNCTIONS.keys())
    #     )
    # )
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="AUTO"
        )
    )


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
    visionqatask=VisionQATask(dataset_path=dataset_path, task_id="sample_task",input_template=MATHVISION_INPUT_TEMPLATE)
    task_goal, task_info= visionqatask.setup()
    INPUT_TEMPLATE= MATHVISION_INPUT_TEMPLATE
    question= "User input: In the picture on the right a number should be written next to each point. The sum of the numbers on the corners of each side of the hexagon should be equal. Two numbers have already been inserted. Which number should be in the place marked '$x$'?\n<image1>"
    options=""
    image_url = "234.jpg"
    text_prompt= INPUT_TEMPLATE.format(question=question, options=options)
    image_input = Image.open(image_url).convert("RGB")
    image_list=[image_input]
    image_path_map = {id(image_input): image_url}
    contents = [text_prompt]
    if image_input:
        contents.append("Observation 0:")
        contents.append(image_input)
    def _stringify_observation_item(item):
        if isinstance(item, Image.Image):
            return {"image_data": image_path_map.get(id(item))}
        if isinstance(item, types.Content):
            parts=item.parts
            listofdict_parts = []
            for part in parts:
                dict_part = {
                    "text": part.text if part.text else None,
                    "thought": part.thought,
                    "function_call": {
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                    } if part.function_call else None,
                    "function_response": {
                        "name": part.function_response.name,
                        "response": part.function_response.response,
                    } if part.function_response else None,
                }
                listofdict_parts.append(dict_part)
            item = {"parts": listofdict_parts, "role": item.role}
        if isinstance(item, str) and item.startswith("parts=") and " role=" in item:
            parts_str, role_part = item.split(" role=", 1)
            role_part = role_part.strip()
            if role_part.startswith("'") and role_part.endswith("'"):
                role_part = role_part[1:-1]
            return {
                "parts": parts_str[len("parts="):],
                "role": role_part,
            }
        return item
    conversation_history = []
    max_tool_calls = 8
    for i in range(max_tool_calls):
        action, extra_info = api_agent.act(contents)
        contents.append(action)
        thinking_process = ""
        final_answer = ""
        function_call_part_list=[]
        for part in action.parts:
            if part.function_call:
                function_call_part_list.append(part)
            elif part.thought:
                thinking_process += part.text
            elif part.text:
                final_answer = part.text
            else:
                continue
        contents_for_save = [_stringify_observation_item(item) for item in contents]
        conversation_history.append({
            "step": i+1,
            "observation": contents_for_save,
            "action": _stringify_observation_item(action),
            "thinking_process": thinking_process,
            "final_answer": final_answer,
            "function_call": [(function_call_part.function_call.name, function_call_part.function_call.args) for function_call_part in function_call_part_list] if function_call_part_list else None,
            "extra_info": extra_info,
        })
    
        if not function_call_part_list or not function_call_part_list[-1].function_call:
            logging.info("Final response generated without further tool calls.")
            break
        dynamic_image = None
        dynamic_image_index = None
        last_function_call = None
        error_result = None
        for function_call_part in function_call_part_list:
            function_call = function_call_part.function_call
            last_function_call = function_call
            logging.info(f"Function to call: {function_call.name}")
            logging.info(f"Arguments: {function_call.args}")
            if function_call.name in TOOL_FUNCTIONS.keys():
                function_to_call = TOOL_FUNCTIONS[function_call.name]
                if dynamic_image is not None and dynamic_image_index is not None:
                    dynamic_image_list = list(image_list)
                    dynamic_image_list[dynamic_image_index] = dynamic_image
                else:
                    dynamic_image_list = image_list
                try:
                    result = function_to_call(dynamic_image_list, **function_call.args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown function {function_call.name}"}
            if isinstance(result, Image.Image):
                dynamic_image = result
                dynamic_image_index = function_call.args.get("image_index", dynamic_image_index)
            else:
                logging.warning(f"Function call failed as {result}")
                dynamic_image = None
                error_result = result
                break
                
        if isinstance(dynamic_image, Image.Image):
            image_list.append(dynamic_image)
            image_name = f"output_{len(image_list)-1}.jpg"
            image_path = os.path.join(image_save_dir, image_name)
            dynamic_image.save(image_path)
            image_path_map[id(dynamic_image)] = image_path
            
            image_bytes = open(image_path, "rb").read()

            function_response_data = {
                "image_ref": {f"Observation {len(image_list)-1}": image_name},
            }
            function_response_multimodal_data = types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(
                    mime_type="image/jpeg",
                    display_name=image_name,
                    data=image_bytes,
                )
            )
            contents.append(
                types.Content(role="tool",parts=[types.Part.from_function_response(
                name=last_function_call.name,
                response=function_response_data,
                parts=[function_response_multimodal_data]
            )])
        )
        else:
            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=last_function_call.name,
                            response={"error": error_result.get("error", "Unknown error")},
                        )
                    ],
                )
            )
        
    else:
        logging.info("Max tool calls reached; forcing final answer without tool calls.")
        FORCE_ANSWER_PROMPT = "Max tool calls reached. Please provide the final answer without further tool calls."
        contents.append(FORCE_ANSWER_PROMPT)
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
        response = api_agent.client.models.generate_content(
            model=api_agent.model,
            contents=contents,
            config=force_generate_config,
        )
        extra_info = {
            "original_response": str(response),
            "tokens_used": response.usage_metadata.total_token_count,
            "model_name": api_agent.config.model_name,
            "attempt": 1,
            "step_count": api_agent.step_count,
        }
        api_agent.step_count += 1
        if extra_info["tokens_used"]:
            api_agent.total_tokens_used += extra_info["tokens_used"]
        action = response.candidates[0].content
        contents.append(action)
        thinking_process = ""
        final_answer = ""
        for part in action.parts:
            if part.thought:
                thinking_process += part.text
            elif part.text:
                final_answer = part.text
        contents_for_save = [_stringify_observation_item(item) for item in contents]
        conversation_history.append({
            "step": len(conversation_history) + 1,
            "action": _stringify_observation_item(action),
            "observation": contents_for_save,
            "thinking_process": thinking_process,
            "final_answer": final_answer,
            "function_call": None,
            "extra_info": extra_info,
        })
    extra_info_list = []
    function_call_total_count = 0
    function_call_each_count = {}
    function_call_per_step = []
    tokens_used_total = 0
    tokens_used_per_step = []
    for record in conversation_history:
        function_call = record.get("function_call")
        if function_call:
            function_call_total_count += len(function_call)
            function_names = []
            for function_name, _ in function_call:
                function_call_each_count[function_name] = function_call_each_count.get(function_name, 0) + 1
                function_names.append(function_name)
            function_call_per_step.append(function_names)
        else:
            function_call_per_step.append(None)
        tokens_used = record.get("extra_info", {}).get("tokens_used", 0)
        tokens_used_total += tokens_used
        tokens_used_per_step.append(tokens_used)
    meta_info = {
        "question": question,
        "options": options,
        "image_path": image_url,
        "function_call_total_count": function_call_total_count,
        "total_steps": len(conversation_history),
        "function_call_each_count": function_call_each_count,
        "function_call_per_step": function_call_per_step,
        "tokens_used_total": tokens_used_total,
        "tokens_used_per_step": tokens_used_per_step,
        "final_answer": conversation_history[-1]["final_answer"] if conversation_history else "",
    }
    last_step_index = len(conversation_history) - 1
    for idx, record in enumerate(conversation_history):
        observation = record.get("observation")
        extra_info_list.append({
            "step": record["step"],
            "extra_info": record.pop("extra_info"),
            "observation": observation,
        })
        if idx != last_step_index:
            record.pop("observation", None)
    
    with open(extra_info_jsonl_path, "w", encoding="utf-8") as f:
        for record in extra_info_list:
            f.write(json.dumps(record) + "\n")

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for record in conversation_history:
            f.write(json.dumps(record) + "\n")
    
    with open(meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta_info) + "\n")
            
if __name__ == "__main__":
    main()
