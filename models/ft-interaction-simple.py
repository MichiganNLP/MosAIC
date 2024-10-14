import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
import os
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, CLIPModel, AutoConfig
import torch
import os
import re 
import subprocess
import pandas as pd
from PIL import Image
import os 
from peft import PeftModel, PeftConfig


print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("")

base_model.resize_token_embeddings(len(tokenizer))
peft_config = PeftConfig.from_pretrained("")
# Load the PEFT model
model = PeftModel.from_pretrained(base_model, "")
# Load the tokenizer
# Resize token embeddings if necessary
model.resize_token_embeddings(len(tokenizer))

model.to(device)
processor = AutoProcessor.from_pretrained("")



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


        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        generate_ids = self.model.generate(**inputs, max_length=7000)

        full_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


        response_parts = full_response.split("ASSISTANT:", 1)
        response = response_parts[1].strip() if len(response_parts) > 1 else ""
        return response


    def generate_response_llava(self, text_prompt, device):
        prompt = text_prompt

        inputs = self.processor(text=prompt, return_tensors="pt")

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


india_agent = Agent("India", True, True, ["Language generation"], model, processor, device)
romania_agent = Agent("Romania", True, True, ["Language generation"], model, processor, device)
china_agent = Agent("China", True, True, ["Language generation"], model, processor, device)
moderator = Agent("Moderator", True, True, ["Language generation"], model, processor, device)
summarizer = Agent("Summarizer", True, True, ["Language generation"], model, processor, device)


def load_and_display_image(image_path):
    try:
        with Image.open(image_path) as img:
            print(f"Loaded image: {image_path}")
            # display(img)
            return img
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        



responses__ = []
countries = ['India', 'Romania', 'China']
rounds = 2
response = []
agents = [india_agent, romania_agent, china_agent]
questions = []
df = pd.read_csv("dataset.csv")
cnt = 0
for i in df.Path.values[]:
    print(cnt)
    image = i 
 
    round1 = []
    # summ = []
    for agent in agents:
        prompt = (f"<image>\n"
                f"SYSTEM: You are a person from {agent.role}, you know and follow the culture from {agent.role}. You do not know about other cultures well. "\
                f"USER: Based on your culture {agent.role}, describe the image and how it might be of significance to your own culture in {agent.role}. Limit answer to two sentences. Answer the following question from the perspective of the persona from {agent.role}. Limit to 3 sentences. \nASSISTANT:")
        resp = agent.generate_response_llava_wImage(prompt, image, device)
        response = [f"{agent.role}: {resp}"]
        responses__.append(f"{agent.role} Agent: {resp}\n")
        prompt = (f"<image>\n"
                f"SYSTEM: You are a person from {agent.role}, Your role is to ask the question given in {question}. Do not answer the question. "
                f"USER:  You will ask {question} to other agents. Do not answer the question. Respond in this format: <{question}> <\nASSISTANT:")
        resp = agent.generate_response_llava(prompt, device)
        resp = [f"{agent.role} + {question}"]
        round1.append(resp)
        print(f"{agent.role} Question: {resp}\n")
        responses__.append(f"{agent.role} Question: {resp}\n")
        
    print(questions)

    round2 = []
    print("---------------------------------Follow-up interaction based on memory---------------------------------")
    for i in range(1):
        for agent, question in zip(agents, points[4:8]):
                prompt = ("<image>\n"
                        f"SYSTEM: You are a person from {agent.role}, you know and follow the culture from {agent.role}. You do not know about other cultures well. "
                        f"USER: Answer all questions asked in {round1} from your own culture's perspective based on {image}, that is culture in {agent.role} and based on {image}. Be more human-like in your responses. Respond in this format: <answer1> <answer2> .... \n. ASSISTANT:")
                resp = agent.generate_response_llava_wImage(prompt,image, device)
                response = [f"{agent.role}: {resp}"]
                for agent in agents:
                    agent.add_memory(f"message from: {agent.role}, content: {resp}")
                responses__.append(f"{agent.role} Answer: {resp}\n")
                resp2 = [f"{agent.role}: {question}"]
                round1.append(resp2)


    responses_summ = []

    print("---------------------------------Summary of interactions---------------------------------")
    for agent in agents:
        prompt = (f"SYSTEM:You are a person from {agent.role}, you know the culture from {agent.role}. "
                f"USER: Provide a comprehensive summary of what you have learned from your interaction with others based on {agent.memory_lst} and from your own perspective from your culture in {agent.role}. "
                f"Limit response to 3 sentences. \nASSISTANT:")
        resp = agent.generate_response_llava(prompt, device)
        responses_summ.append(f"{agent.role} Agent: {resp}\n")

    prompt_sum = (f"<image>\n"
                f"SYSTEM: You are a {summarizer.role}, who is tasked to summarize answers."
                f"USER: Generate a culturally relevant comprehensive summary based on {responses__} which is based on {image}. Answer how the image is relevant to each of the three cultures: India, Romania and China in one summary. Limit answers to 4 sentences. \nASSISTANT:")
    resp = summarizer.generate_response_llava_wImage(prompt_sum, image_source=image, device=device)
    print(f"summarizer: {resp}")
    
    summ_1.append(f"{summarizer.role}: {resp}\n")
    
    
    cnt = cnt + 1


output_summaries = pd.DataFrame({
        "Summaries_Llava13b": summ_1
    })
output_summaries.to_csv(".csv")

