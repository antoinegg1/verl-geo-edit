from typing import Tuple, Dict, Any, List, Optional
import io
from google.genai import types
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .base import AbstractVLMTask

class VisionQATask(AbstractVLMTask):
    """vision qa task"""
    
    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str,
        save_dir: Path | str,
        tool_functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(task_id)
        self.task_prompt = task_prompt
        self.task_answer = task_answer
        self.task_image_path = task_image_path
        self.tool_functions = tool_functions 
        self.options = kwargs.get("options", None)
        
        
        self.image_path_map : Dict[int, str] = {}
        self.image_list=[Image.open(self.task_image_path).convert("RGB")]
        
        self.contents=[self.task_prompt]
        self.conversation_history: List[Dict[str, Any]] = []
        
        self.contents.append("Observation 0:")
        self.contents.append(self.image_list[0])
        
        self.save_dir=save_dir
        os.makedirs(self.save_dir, exist_ok=True) 
        self.output_jsonl_path=os.path.join(self.save_dir, "output.jsonl")
        self.extra_info_jsonl_path=os.path.join(self.save_dir, "extra_info.jsonl")
        self.meta_info_jsonl_path=os.path.join(self.save_dir, "meta_info.jsonl")
        self.image_save_dir=os.path.join(self.save_dir, "images")
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        

    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task"""
        return 0.0, False, {}
    
    def get_info(self) -> Dict[str, Any]:
       pass

       
    def _stringify_observation_item(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, Image.Image):
            return {"image_data": self.image_path_map.get(id(item))}
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
    
    def parse_action(self, step: int, action: types.Content, extra_info: Dict[str, Any]):
        """update task contents from action"""
        self.contents.append(action)
        thinking_process = ""
        final_answer = ""
        function_call_part_list = []
        for part in action.parts:
            if part.thought:
                thinking_process += part.text
            elif part.function_call:
                function_call_part_list.append(part)
            elif part.text:
                final_answer = part.text
            else:
                continue
        contents_for_save=[self._stringify_observation_item(item) for item in self.contents]
        self.conversation_history.append({
            "step": step,
            "observation": contents_for_save,
            "action": self._stringify_observation_item(action),
            "thinking_process": thinking_process,
            "final_answer": final_answer,
            "function_call": [(function_call_part.function_call.name, function_call_part.function_call.args) for function_call_part in function_call_part_list] if function_call_part_list else None,
            "extra_info": extra_info,
        })
        
        return function_call_part_list
    
    def update_observation_from_action(self, function_call_part_list: Any):
        
        dynamic_image=None
        dynamic_image_index=None
        last_success_function_call=None
        error_result=[]
        for function_call_part in function_call_part_list:
            function_call=function_call_part.function_call
            logging.info(f"Processing function call: {function_call.name} with args: {function_call.args}")
            if function_call.name in self.tool_functions.keys():
                function_to_call=self.tool_functions[function_call.name]
                if dynamic_image is not None and dynamic_image_index is not None:
                    dynamic_image_list = list(self.image_list)
                    dynamic_image_list[dynamic_image_index] = dynamic_image
                else:
                    dynamic_image_list = self.image_list
                try:
                    result=function_to_call(dynamic_image_list, **function_call.args)
                    
                    dynamic_image = result
                    dynamic_image_index = function_call.args.get("image_index", dynamic_image_index)
                    last_success_function_call = function_call
                    
                except Exception as e:
                    result = {"function_name": function_call.name, "error_msg":f"Function call {function_call.name} with args {function_call.args} failed with error: {str(e)}"}
                    logging.warning(f"Function call failed as {result}")
                    error_result.append(result)
            else:
                result = {"function_name": function_call.name, "error_msg":f"Unknown function {function_call.name}"}    
                logging.warning(f"Function call failed as {result}")
                error_result.append(result)

        if isinstance(dynamic_image, Image.Image):
            self.image_list.append(dynamic_image)
            image_name=f"output_{len(self.image_list)-1}.jpg"
            image_path=os.path.join(self.image_save_dir, image_name)
            dynamic_image.save(image_path)
            self.image_path_map[id(dynamic_image)] = image_path
            image_bytes_io = io.BytesIO()
            dynamic_image.save(image_bytes_io, format="JPEG")
            image_bytes = image_bytes_io.getvalue()
            
            function_response_data = {
                "image_ref": {f"Observation {len(self.image_list)-1}": image_name},
            }
            function_response_multimodal_data = types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(
                    mime_type="image/jpeg",
                    display_name=image_name,
                    data=image_bytes,
                )
            )
            self.contents.append(
                types.Content(role="tool",
                              parts=[
                                  types.Part.from_function_response(
                                        name=last_success_function_call.name,
                                        response=function_response_data,
                                        parts=[function_response_multimodal_data]
                                        )
                                    ]
                              )
                )
        else:
            for err in error_result:
                self.contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=err["function_name"],
                                response={"error": err["error_msg"]}
                            )
                        ]
                    )
                )
        
    def save_trajectory(self):
        """save the trajectory to jsonl files"""
        extra_info_list = []
        function_call_total_count = 0
        function_call_each_count = {}
        function_call_per_step = []
        tokens_used_total = 0
        tokens_used_per_step = []
        
        for record in self.conversation_history:
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
            "question": self.task_prompt,
            "options": self.options,
            "image_path": self.task_image_path,
            "function_call_total_count": function_call_total_count,
            "total_steps": len(self.conversation_history),
            "function_call_each_count": function_call_each_count,
            "function_call_per_step": function_call_per_step,
            "tokens_used_total": tokens_used_total,
            "tokens_used_per_step": tokens_used_per_step,
            "final_answer": self.conversation_history[-1]["final_answer"] if self.conversation_history else "",
        }
        
        last_step_index = len(self.conversation_history) - 1
        for idx, record in enumerate(self.conversation_history):
            observation = record.get("observation")
            extra_info_list.append({
                "step": record["step"],
                "extra_info": record.pop("extra_info"),
                "observation": observation,
            })
            if idx != last_step_index:
                record.pop("observation", None)
        
        with open(self.extra_info_jsonl_path, "w", encoding="utf-8") as f:
            for record in extra_info_list:
                f.write(json.dumps(record) + "\n")
        
        with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
            for record in self.conversation_history:
                f.write(json.dumps(record) + "\n")
        
        with open(self.meta_info_jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_info) + "\n")
            
