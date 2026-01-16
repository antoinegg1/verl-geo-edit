from google import genai
import io
from google.genai import types
from PIL import Image
import requests
from ..action import TOOL_FUNCTIONS, TOOL_FUNCTIONS_DECLARE
from ..config import API_KEY
from ..constants import SYSTEM_PROMPT,MATHVISION_INPUT_TEMPLATE

INPUT_TEMPLATE= MATHVISION_INPUT_TEMPLATE


# Configure the client and tools
client = genai.Client(api_key=API_KEY)
tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
tool_config = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=list(TOOL_FUNCTIONS.keys())
    )
)

config = types.GenerateContentConfig(
    tools=[tools],
    thinking_config=types.ThinkingConfig(
        thinkingLevel="low",
        include_thoughts=True
    ),
    temperature=1.0,
    system_instruction=[SYSTEM_PROMPT],
    max_output_tokens=10240,
    candidate_count=1,
)

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


max_tool_calls = 8
for i in range(max_tool_calls):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=config,
    )
    content = response.candidates[0].content
    function_call_part = None
    for part in content.parts:
        if part.function_call:
            function_call_part = part
            break
    if not function_call_part or not function_call_part.function_call:
        print("No function call found in the response.")
        print(response)
        break
    function_call = function_call_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    if function_call.name in TOOL_FUNCTIONS.keys():
        function_to_call = TOOL_FUNCTIONS[function_call.name]
        result = function_to_call(image_list, **function_call.args)
    else:
        result = {"error": f"Unknown function {function_call.name}"}
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"Function call: {function_call}\n")
        f.write(f"Function result: {result}\n\n")
    if isinstance(result, Image.Image):
        image_list.append(result)
        image_name = f"output_{len(image_list)-1}.jpg"
        result.save(image_name)
        
        image_bytes = open(image_name, "rb").read()
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

    contents.append(response.candidates[0].content)
    contents.append(
        types.Content(role="tool",parts=[types.Part.from_function_response(
          name=function_call.name,
          response=function_response_data,
          parts=[function_response_multimodal_data]
        )])
    )
else:
    print("Max tool calls reached without final response.")