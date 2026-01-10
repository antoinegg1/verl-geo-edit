from .image_edition_nanob import image_edition_function
image_edit_function_declaration = {
    "name":"image_edition",
    "description":'''
    Calling an image editing tool with proper prompt and existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to edit the image as instructions. Returns the edited image.
    For example, to remove an object from the image, you can provide instructions like 'Remove the red car from the image' along with the appropriate image index; to change colors, you can say 'Change the sky to a sunset orange' etc.
    REMEMBER that this function can ONLY edit images without any reasoning or calculations, so please provide clear and concise instructions on how to edit the image.
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to be edited. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "prompt":{
                "type":"string",
                "description":"Instructions on how to edit the image."
            }
        },
        "required":["image_index", "prompt"]
    }
}

TOOL_FUNCTIONS_DECLARE = [image_edit_function_declaration]
TOOL_FUNCTIONS={
    "image_edition":image_edition_function,
}

