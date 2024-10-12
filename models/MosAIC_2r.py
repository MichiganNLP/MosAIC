import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
import os
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, CLIPModel
import torch
import os
import re 

current_path = os.getcwd()
print("Current path:", current_path)

# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/datasets/GD-VCR/GD_VCR.csv'
# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/datasets/dollarstreet_unprocessed/dollarstreet_RIC_tagged.csv'
file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/datasets/CVQA/cvqa_RIC_tagged.csv'
# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/datasets/GeoDE/divided/GeoDE_0.csv'
df = pd.read_csv(file_path)


print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16)
model.to(device)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")


class Agent:
    def __init__(self, role, can_create_agents, can_halt_agents, plugins, model, processor, device):
        self.conversation_history = []
        self.role = role
        self.can_create_agents = can_create_agents
        self.can_halt_agents = can_halt_agents
        self.plugins = plugins
        self.state = None
        self.memory_lst = []

        self.model = model
        self.processor = processor
        self.device = device

    def generate_response_llava_wImage(self, text_prompt, image_source, device):
        image = Image.open(image_source)
        prompt = text_prompt
        # #######
        # if image is None:
        #     raise ValueError(f"Failed to load image from {image_source}")
        # else:
        #     print("Image successfully loaded.")
        # #######

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # #######
        # print("Processed inputs:", inputs)

        # for key, value in inputs.items():
        #     print(f"{key}: shape {value.shape}, type {value.dtype}")
        # #######

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # #######
        # print("Inputs being sent to model:", inputs)
        # #######

        generate_ids = self.model.generate(**inputs, max_length=100000)

        full_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


        response_parts = full_response.split("ASSISTANT:", 1)
        response = response_parts[1].strip() if len(response_parts) > 1 else ""
        return response


    def generate_response_llava(self, text_prompt, device):
        prompt = text_prompt

        inputs = self.processor(text=prompt, return_tensors="pt")

        # -print("Inputs being sent to model:", inputs)

        # if inputs.get('pixel_values') is None:
        #     del inputs['pixel_values']

        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

        generate_ids = self.model.generate(**inputs, max_length=100000)

        full_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        

        response_parts = full_response.split("ASSISTANT:", 1)
        response = response_parts[1].strip() if len(response_parts) > 1 else ""
        return response


    def create_agent(self, role, can_create_agents, can_halt_agents, plugins, model, processor):
        if self.can_create_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to create new agents")

    def halt_agent(self):
        if self.can_halt_agents:
            self.state = "halted"
        else:
            raise Exception("This agent does not have the ability to halt agents")

    def add_event(self, event):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory):
        self.memory_lst.append({"role": "assistant", "content": memory})


# device = torch.device("cuda")
# model.to(device)

# usa_agent = Agent("USA", True, True, ["Language generation"], model, processor, device)
india_agent = Agent("India", True, True, ["Language generation"], model, processor, device)
romania_agent = Agent("Romania", True, True, ["Language generation"], model, processor, device)
china_agent = Agent("China", True, True, ["Language generation"], model, processor, device)
moderator = Agent("Moderator", True, True, ["Language generation"], model, processor, device)
summarizer = Agent("Summarizer", True, True, ["Language generation"], model, processor, device)


#  image = Image.open('../2024_embeding_difference/test_images_complexScene/n1.jpeg')

# image_source = "./2024_embeding_difference/test_images_complexScene/Christmas-Celebration-in-India.png"
# image_source = "india2.png"

# base_path = '/nfs/turbo/coe-mihalcea/shared_data/GD-VCR/X_VCR'
# base_path = '/nfs/turbo/coe-mihalcea/shared_data/GeoDE/images'
# base_path = '/nfs/turbo/coe-mihalcea/shared_data/dollarstreet'
base_path = '/nfs/turbo/coe-mihalcea/shared_data/cvqa/data/saved_images_RIC'

def generate_caption(relative_path):


    image_source = os.path.join(base_path, relative_path)

    responses__ = []
    countries = ['India', 'Romania', 'China']
    rounds = 2
    response = []
    agents = [india_agent, romania_agent, china_agent]


    #### Multi-agent interaction round 1 start 
    round1 = []

    for agent in agents:
        prompt = (f"<image>\n"
                f"SYSTEM: You are a person from {agent.role}, you know the culture from {agent.role} pretty well, but you don't have too much knowledge for other cultures. You as a human from {agent.role} always generate conversational language in human-like dialogue style"
                f"USER: Remember you are from {agent.role}, first observe the image and think what you see in the image; \
                    then think of how the object you saw is related to your culture in {agent.role}; \
                    finally, generate human-like conversational language to describe the object you saw in the image and how is this related to your culture in {agent.role}. \
                    Remember to be conversational and in human-like dialogue style. Limit answer to 3 sentences. \nASSISTANT:")
                # f"USER: Based on your culture {agent.role}, describe what you first see in the image based on its significance to your culture in {agent.role}. Limit answer to two sentences. \nASSISTANT:")
        resp = agent.generate_response_llava_wImage(prompt, image_source, device)
        response = [f"{agent.role}: {resp}"]
        for agent_mem in agents:
            agent_mem.add_memory(f"message from: {agent.role}, content: {resp}")

        ######### test
        #print('prompt:', prompt)
        ##############

        # print(f"{agent.role}: {resp}\n")
        responses__.append(f"{agent.role} Agent: {resp}\n")


    ##########################
    summary__ = []
    # print("---------------------------------Summary of interactions---------------------------------")
    for agent in agents:
        prompt = (f"USER: First read the conversation history {responses__} among different people, understand this as a discussion about the image and the culture among people; \
                    then find the contents in the conversation that related to the image contents description, and the culture related discussion; \
                    finally provide a summary of what you have learned from the image contents description and the culture related discussion, do the summary from your perspective as a person from {agent.role}. "
                f"Answer with 3 sentences to give a detailed summary. \nASSISTANT:")
        resp = agent.generate_response_llava(prompt, device)
        ######### test
        #print('prompt:', prompt)
        ##############
        # print(f"{agent.role} Summary: {resp}\n")
        summary__.append(f"{agent.role} Agent: {resp}\n")

    ## Sumarizer 
    prompt_sum = (f"<image>\n"
                f"SYSTEM: You are a {moderator.role}, who is tasked to summarize answers."
                f"USER: Given the conversation history: {summary__} and the image, \
                    first read the conversation history and understand this as a summary from each people in a discussion about the image description and the related cultures; \
                    next, from the conversation history, find what contents are about the image content description\
                    then, from the conversation history, find what contents are about the image related cultura knowledge; \
                    finally, generate a comprehensive summary based on the conversation history contents and the image: describe the content of the picture in the first sentence, and then describe the cultural knowledge related to the picture after that. \
                    Answer in this format: <summary>. Limit response to 3 sentences.  \nASSISTANT:")
    resp = summarizer.generate_response_llava_wImage(prompt_sum, image_source=image_source, device=device)

    caption = resp

    return caption




df['llava_agent_caption'] = [generate_caption(path) for path in tqdm(df['image_path'], desc="Generating Captions")] #  dollarstreet: imageRelPath; GD-VCR: img_path

# Save the DataFrame with captions
output_file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/caption_results/ablation1r_correct/llava_agent_cot_ablation1r_CVQA.csv'
df.to_csv(output_file_path, index=False)

print("Captions generated and saved successfully.")



# output_df = pd.DataFrame({
#     "responses": responses__
# })


# pd.DataFrame(responses__).to_csv("testOutputs.csv")
