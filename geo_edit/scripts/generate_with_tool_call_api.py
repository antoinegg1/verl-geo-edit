from google import genai
import io
from google.genai import types
from PIL import Image
from ..action import TOOL_FUNCTIONS, TOOL_FUNCTIONS_DECLARE
from ..config import API_KEY

SYSTEM_PROMPT = '''
You are an advanced AI agent capable of complex
reasoning and tool usage. You must strictly adhere
to the following protocol for every interaction:
1. ALWAYS call the appropriate tool first;
2. NEVER provide answers without tool results;
3. Call appropriate tools based on the task;
4. Provide clear and concise instructions to tools, NEVER ask tool to directly solve the problem;
5. Reasoning Before Action: before selecting a tool,
you must analyze the user’s request and determine
the necessary steps. Output your internal monologue
and logic inside <think> and </think> tags.
6. Reasoning After Action: Once you receive the
output from a tool, you must analyze the results to
determine if further actions are needed or if the task
is complete. Output this analysis inside <think> and
</think> tags;
7. Final Output: When you have formulated your
conclusion, you must wrap your final answer in
<answer> and </answer> tags.
'''
INPUT_TEMPLATE = "Please solve the problem with provided tools. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}"

# Configure the client and tools
client = genai.Client(api_key=API_KEY)
tools = types.Tool(function_declarations=TOOL_FUNCTIONS_DECLARE)
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

question= "User input: How many bricks are missing in the wall?\n"
options=""
image_url = "7.jpg"
text_prompt= INPUT_TEMPLATE.format(question=question, options=options)
image_input = Image.open(image_url).convert("RGB")
image_list=[image_input]
contents = [text_prompt]
if image_input:
    contents.append("Observation 0:")
    contents.append(image_input)

#TODO: build up a recycle loop to call tools until final answer is found,add contents by contents.append(response.candidates[0].content) # Append the content from the model's response. contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response



# Send request with function declarations
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
# Check for a function call
if function_call_part and function_call_part.function_call:
    function_call = function_call_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #  In a real app, you would call your function here:
    #  result = schedule_meeting(**function_call.args)
    if function_call.name in TOOL_FUNCTIONS.keys():
        function_to_call = TOOL_FUNCTIONS[function_call.name]
        result = function_to_call(image_list, **function_call.args)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"Function call: {function_call}\n")
        f.write(f"Function result: {result}\n\n")
    if isinstance(result, Image.Image):      
        result.save("edited_image.png")
    else:
        print(f"Function result: {result}")
        
else:
    print("No function call found in the response.")
    print(response)