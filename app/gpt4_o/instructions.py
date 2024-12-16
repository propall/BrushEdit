def create_editing_category_messages_gpt4o(editing_prompt):
    messages = [{
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will give you an editing instruction of the image. Please output which type of editing category it is in. You can choose from the following categories: \n\
                    1. Addition: Adding new objects within the images, e.g., add a bird \n\
                    2. Remove: Removing objects, e.g., remove the mask \n\
                    3. Local: Replace local parts of an object and later the object's attributes (e.g., make it smile) or alter an object's visual appearance without affecting its structure (e.g., change the cat to a dog) \n\
                    4. Global: Edit the entire image, e.g., let's see it in winter \n\
                    5. Background: Change the scene's background, e.g., have her walk on water, change the background to a beach, make the hedgehog in France, etc. \n\
                    Only output a single word, e.g., 'Addition'.",
                },]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
            ]
            }]
    return messages
    

def create_ori_object_messages_gpt4o(editing_prompt):

    messages =  [
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "I will give you an editing instruction of the image. Please output the object needed to be edited. You only need to output the basic description of the object in no more than 5 words.  The output should only contain one noun. \n \
                    For example, the editing instruction is 'Change the white cat to a black dog'. Then you need to output: 'white cat'. Only output the new content. Do not output anything else."
                    },]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": editing_prompt
                    }
                ]
                }
            ]
    return messages


def create_add_object_messages_gpt4o(editing_prompt, base64_image, height=640, width=640):

    size_str = f"The image size is height {height}px and width {width}px. The top - left corner is coordinate [0 , 0]. The bottom - right corner is coordinnate [{height} , {width}]. "

    messages = [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "I need to add an object to the image following the instruction: " + editing_prompt + ". " + size_str + " \n \
                    Can you give me a possible bounding box of the location for the added object? Please output with the format of [top - left x coordinate , top - left y coordinate , box width , box height]. You should only output the bounding box position and nothing else. Please refer to the example below for the desired format.\n\
                    [Examples]\n \
                    [19, 101, 32, 153]\n  \
                    [54, 12, 242, 96]"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                        ]
                        }
                    ]
    return messages


def create_apply_editing_messages_gpt4o(editing_prompt, base64_image):
    messages =  [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "I will provide an image along with an editing instruction. Please describe the new content that should be present in the image after applying the instruction. \n \
                    For example, if the original image content shows a grandmother wearing a mask and the instruction is 'remove the mask', your output should be: 'a grandmother'. The output should only include elements that remain in the image after the edit and should not mention elements that have been changed or removed, such as 'mask' in this example. Do not output 'sorry, xxx', even if it's a guess, directly output the answer you think is correct."
                },]
            },      
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": editing_prompt
                },
                {"type": "image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                }, 
            ]
            }
        ]
    return messages
