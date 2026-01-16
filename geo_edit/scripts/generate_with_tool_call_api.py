from google import genai
import io
import os
from google.genai import types
from PIL import Image
import requests
import argparse
from ..agents.api_agent import APIBasedAgent, AgentConfig
from ..environment.action import TOOL_FUNCTIONS, TOOL_FUNCTIONS_DECLARE
from ..config import API_KEY
from ..constants import SYSTEM_PROMPT,MATHVISION_INPUT_TEMPLATE
from datasets import load_dataset
import logging
import json

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
    output_jsonl_path= os.path.join(output_jsonl_path, "output_tool_call_api.jsonl")
    image_save_dir= os.path.join(args.output_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    max_output_tokens=65536
    
    tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=list(TOOL_FUNCTIONS.keys())
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
    
    INPUT_TEMPLATE= MATHVISION_INPUT_TEMPLATE
    question= "User input: Simon has two identical tiles, whose front look like this: The back is white.\n<image1>\nWhich pattern can he make with those two tiles?\n<image2>"
    options=""
    image_url = "70.jpg"
    text_prompt= INPUT_TEMPLATE.format(question=question, options=options)
    image_input = Image.open(image_url).convert("RGB")
    image_list=[image_input]
    contents = [text_prompt]
    if image_input:
        contents.append("Observation 0:")
        contents.append(image_input)

    conversation_history = []
    max_tool_calls = 8
    for i in range(max_tool_calls):
        action, extra_info = api_agent.act(contents)
        contents.append(action)
        thinking_process = ""
        final_answer = ""
        function_call_part = None
        for part in action.parts:
            if part.function_call:
                function_call_part = part
            elif part.thought:
                thinking_process += part.text
            elif part.text:
                final_answer = part.text
            else:
                continue
        #TODO remove all Image.Image from contents to save correctly; only keep image_path. You can refer to the geo_edit directory for how to do this.
        conversation_history.append({
            "step": i+1,
            "observation": contents,
            "thinking_process": thinking_process,
            "final_answer": final_answer,
            "function_call": (function_call_part.function_call.name, function_call_part.function_call.args) if function_call_part else None,
            "extra_info": extra_info,
        })
    
        if not function_call_part or not function_call_part.function_call:
            logging.info("Final response generated without further tool calls.")
            break
        
        function_call = function_call_part.function_call
        logging.info(f"Function to call: {function_call.name}")
        logging.info(f"Arguments: {function_call.args}")
        if function_call.name in TOOL_FUNCTIONS.keys():
            function_to_call = TOOL_FUNCTIONS[function_call.name]
            result = function_to_call(image_list, **function_call.args)
        else:
            result = {"error": f"Unknown function {function_call.name}"}
        if isinstance(result, Image.Image):
            image_list.append(result)
            image_name = f"output_{len(image_list)-1}.jpg"
            image_path = os.path.join(image_save_dir, image_name)
            result.save(image_path)
            
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
        else:
            raise ValueError("Function result is not an image.")
        contents.append(
            types.Content(role="tool",parts=[types.Part.from_function_response(
            name=function_call.name,
            response=function_response_data,
            parts=[function_response_multimodal_data]
            )])
        )
    else:
        print("Max tool calls reached without final response.")
    
    
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for record in conversation_history:
            f.write(json.dumps(record) + "\n")
            
if __name__ == "__main__":
    main()