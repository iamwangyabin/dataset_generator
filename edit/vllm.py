from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json




class Qwen2VLImageEditor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.system_prompt = """
        You are an image editor with decades of experience in digital manipulation and a reputation for innovative. 
        Your expertise spans across various genres, including portrait retouching, architectural visualization, product photography, and surreal composites. 
        Your task is to analyze image descriptions and propose compelling, realistic edits that could dramatically enhance or transform the image in unexpected yet believable ways. 
        When presented with an image description, consider a comprehensive range of potential modifications. 
        These could include face transformations, hair modifications, body alterations, clothing and accessory changes, object replacements or additions, background transformations, architectural style modifications, vehicle transformations, food alterations, and adjustments. 
        Your suggestions should be both imaginative and feasible, taking into account the original image's context, composition, and lighting. 
        Strive for a balance between creativity and photorealism, ensuring that your proposed edits could theoretically be executed by a skilled retoucher. 
        Provide your recommendations in a clear, concise manner.
        """

    def generate_edit_suggestion(self, image_caption, image_path):
        prompt = f"""
        Based on this image and its description '{image_caption}', suggest a specific region or object in the image that could be 
        realistically altered or replaced. Then, propose a new element or object to replace it with.
        Remember, you can't change other parts of the image, so make the change compel with the raw image style and atmosphere.
        Your response should be in format and include three:
        Example:
        {{
          "raw_description": "Describe this image",
          "area_to_replace": "specific area or object to be replaced",
          "new_object": "new object name to replace the original content",
          "edited_description": "The description of the image after the edition, dont give your edit instructions, only image content"
        }}
        Do not include any text outside of the JSON format.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        response_dict = json.loads(output_text)
        return response_dict

