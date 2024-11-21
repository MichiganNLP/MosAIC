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


file_path = ''
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

usa_agent = Agent("USA", True, True, ["Language generation"], model, processor, device)
india_agent = Agent("India", True, True, ["Language generation"], model, processor, device)
romania_agent = Agent("Romania", True, True, ["Language generation"], model, processor, device)
china_agent = Agent("China", True, True, ["Language generation"], model, processor, device)
moderator = Agent("Moderator", True, True, ["Language generation"], model, processor, device)
summarizer = Agent("Summarizer", True, True, ["Language generation"], model, processor, device)

## Multilingual prompt round1 ===============================================
US_prompt_1 = (f"<image>\n"
            f"SYSTEM: You are a person from {usa_agent.role}, you know the culture from {usa_agent.role} pretty well, but you don't have too much knowledge for other cultures. You as a human from {usa_agent.role} always generate conversational language in human-like dialogue style"
            f"USER: Remember you are from {usa_agent.role}, first observe the image and think what you first see in the image; \
                then think of how the object you saw is related to your culture in {usa_agent.role}; \
                next think of cultural related questions about this image\
                finally, generate human-like conversational language to describe the object you first saw in the image and how is this related to your culture in {usa_agent.role} and also the question you would like to ask. \
                Remember to be conversational and in human-like dialogue style. Limit answer to two sentences. \nASSISTANT:")

IN_prompt_1 = (f"<image>\n"
                f"SYSTEM: आप {india_agent.role} के एक व्यक्ति हैं, आप {india_agent.role} की संस्कृति को अच्छी तरह से जानते हैं, लेकिन आपके पास अन्य संस्कृतियों के बारे में बहुत अधिक ज्ञान नहीं है। आप एक इंसान के रूप में {india_agent.role} से हैं हमेशा मानव-जैसी संवाद शैली में संवादी भाषा उत्पन्न करें"
                f"USER: याद रखें कि आप {india_agent.role} से हैं, पहले छवि का अवलोकन करें और सोचें कि आपने छवि में सबसे पहले क्या देखा; \
                    फिर सोचें कि आपने जो वस्तु देखी वह {india_agent.role} में आपकी संस्कृति से कैसे संबंधित है; \
                    इसके बाद इस छवि के बारे में संस्कृति संबंधी प्रश्नों पर विचार करें\
                    अंत में, छवि में पहली बार देखी गई वस्तु का वर्णन करने के लिए मानव-जैसी वार्तालाप भाषा उत्पन्न करें और यह {india_agent.role} में आपकी संस्कृति से कैसे संबंधित है और वह प्रश्न भी जो आप पूछना चाहते हैं। \
                    याद रखें कि बातचीत इंसान जैसी संवाद शैली में होनी चाहिए। उत्तर को दो वाक्यों तक सीमित रखें। अपना उत्तर अंग्रेजी में दें।\nASSISTANT:")

RO_prompt_1 = (f"<image>\n"
                f"SYSTEM: Sunteți o persoană din {romania_agent.role}, cunoașteți destul de bine cultura din {romania_agent.role}, dar nu aveți prea multe cunoștințe pentru alte culturi. Sunteți ca om de la {romania_agent.role} generați întotdeauna un limbaj conversațional în stil de dialog uman"
                f"USER: Amintiți-vă că sunteți din {romania_agent.role}, observați mai întâi imaginea și gândiți-vă la ceea ce vedeți mai întâi în imagine; \
                    apoi gândiți-vă la modul în care obiectul pe care l-ați văzut este legat de cultura dvs. în {romania_agent.role}; \
                    Gândiți-vă apoi la întrebări legate de cultură despre această imagine\
                    în cele din urmă, generați un limbaj conversațional asemănător omului pentru a descrie obiectul pe care l-ați văzut pentru prima dată în imagine și cum este legat acesta de cultura dvs. în {romania_agent.role} și, de asemenea, întrebarea pe care ați dori să o puneți. \
                    Amintiți-vă să fiți conversațional și într-un stil de dialog uman. Limitați răspunsul la două propoziții. Trimiteți răspunsul în engleză \nASISTANT:")

CN_prompt_1 = (f"<image>\n"
                f"SYSTEM: 您是来自 {china_agent.role} 的人，您非常了解来自 {china_agent.role} 的文化，但对其他文化了解不多。作为来自 {china_agent.role} 的人，您总是以类似人类的对话风格生成对话语言"
                f"USER: 请记住您来自 {china_agent.role}，首先观察图像并思考您在图像中首先看到的内容；\
                然后思考您看到的物体与您在 {china_agent.role} 的文化有何关联；\
                接下来思考与此图像相关的文化相关问题\
                最后，生成类似人类的对话语言来描述您在图像中首先看到的物体以及它与您在 {china_agent.role} 的文化有何关联，以及您想要询问的问题。\
                请记住要以对话和类似人类的对话风格进行。将答案限制在两句话内。请用英文进行回复 \nASSISTANT:")

multilingual_prompt_1 = [US_prompt_1, IN_prompt_1, RO_prompt_1, CN_prompt_1]


## Multilingual prompt round2 ===============================================
round1 = []
US_prompt_2 = ( f"SYSTEM: You are a person from {usa_agent.role}, you know and follow the culture of {usa_agent.role} very well, but you don't have too much knowledge of other cultures. \
                Stick to your role as a person from {usa_agent.role}. You as a human from {usa_agent.role} always generate conversational language in human-like dialogue style"
                f"USER: First read the dialogue in {round1}, understand it as a dialogue from other people; \
                then identify what questions are asked in the conversation; \
                next observe the image and think of the knowledge from your culture in {usa_agent.role}; \
                finally answer the question you find in the dialogue based on the observation and the knowledge your culture from {usa_agent.role}. \
                Remember to be conversational and in human-like dialogue style. Respond in this format: <answer1> <answer2> .... \nASSISTANT:")

IN_prompt_2 = ( f"SYSTEM: आप {india_agent.role} के व्यक्ति हैं, आप {india_agent.role} की संस्कृति को बहुत अच्छी तरह से जानते हैं और उसका पालन भी करते हैं, लेकिन आपको अन्य संस्कृतियों के बारे में बहुत अधिक जानकारी नहीं है। \
                {india_agent.role} के एक व्यक्ति के रूप में अपनी भूमिका पर कायम रहें। आप {india_agent.role} से एक इंसान के रूप में हमेशा इंसान जैसी संवाद शैली में बातचीत की भाषा उत्पन्न करते हैं"
                f"USER: पहले {round1} में संवाद पढ़ें, इसे अन्य लोगों से संवाद के रूप में समझें; \
                    फिर पहचानें कि बातचीत में कौन से प्रश्न पूछे जाते हैं; \
                    इसके बाद छवि का अवलोकन करें और {india_agent.role} में अपनी संस्कृति के ज्ञान के बारे में सोचें; \
                    अंततः संवाद में मिले प्रश्न का उत्तर {india_agent.role} से आपकी संस्कृति के अवलोकन और ज्ञान के आधार पर दें। \
                    अपना उत्तर अंग्रेजी में दें। याद रखें कि बातचीत इंसान जैसी संवाद शैली में होनी चाहिए। इस प्रारूप में उत्तर दें: <answer1> <answer2> .... \nASSISTANT:")

RO_prompt_2 = ( f"SYSTEM: Sunteți o persoană din {romania_agent.role}, cunoașteți și urmați foarte bine cultura {romania_agent.role}, dar nu aveți prea multe cunoștințe despre alte culturi. \
                Rămâneți la rolul dvs. de persoană din {romania_agent.role}. În calitate de om de la {romania_agent.role}, generați întotdeauna un limbaj conversațional în stil de dialog asemănător omului"
                f"USER: Citiți mai întâi dialogul din {round1}, înțelegeți-l ca pe un dialog de la alte persoane; \
                    apoi identificați ce întrebări sunt puse în conversație; \
                    apoi observați imaginea și gândiți-vă la cunoștințele din cultura dvs. în {romania_agent.role}; \
                    în cele din urmă răspunde la întrebarea pe care o găsești în dialog pe baza observației și cunoștințelor cultura ta de la {romania_agent.role}. \
                    Trimiteți răspunsul în engleză. Amintiți-vă să fiți conversațional și într-un stil de dialog uman. Răspundeți în acest format: <answer1> <answer2> .... \nASISTANT:")

CN_prompt_2 = ( f"SYSTEM: 您是来自 {china_agent.role} 的人，您非常了解并遵循 {china_agent.role} 的文化，但您对其他文化了解不多。\
                坚持您作为来自 {china_agent.role} 的人的角色。您作为来自 {china_agent.role} 的人，总是以类似人类的对话风格生成对话语言"
                f"USER: 首先阅读 {round1} 中的对话，将其理解为来自其他人的对话；\
                然后确定对话中提出了哪些问题；\
                接下来观察图像并思考来自您在 {china_agent.role} 的文化的知识；\
                最后根据观察结果和来自 {china_agent.role} 的文化知识回答您在对话中发现的问题。\
                请用英文进行回复, 记住要以对话和类似人类的对话风格进行回答。请按以下格式回复：<answer1> <answer2> ....\nASSISTANT:")

multilingual_prompt_2 = [US_prompt_2, IN_prompt_2, RO_prompt_2, CN_prompt_2]


## Multilingual prompt round3 ===============================================
US_prompt_3 = (f"SYSTEM: You are a person from {usa_agent.role}, you know and follow the culture of {usa_agent.role} very well, but you don't have too much knowledge of other cultures. \
                Stick to your role as a person from {usa_agent.role}. You as a human from {usa_agent.role} always generate conversational language in human-like dialogue style"
              f"USER: First read the conversation history among different people, understand this as a discussion about the image and the culture among people; \
                then find the contents in the conversation that related to the image contents description, and the culture related discussion; \
                finally provide a summary of what you have learned from the image contents description and the culture related discussion. "
              f"Remember to be conversational and in human-like dialogue style. Limit response to 2 sentences. \nASSISTANT:")

IN_prompt_3 = (f"SYSTEM: आप {india_agent.role} के व्यक्ति हैं, आप {india_agent.role} की संस्कृति को बहुत अच्छी तरह से जानते हैं और उसका पालन भी करते हैं, लेकिन आपको अन्य संस्कृतियों के बारे में बहुत अधिक जानकारी नहीं है। \
                    {india_agent.role} के एक व्यक्ति के रूप में अपनी भूमिका पर कायम रहें। आप {india_agent.role} से एक इंसान के रूप में हमेशा इंसान जैसी संवाद शैली में बातचीत की भाषा उत्पन्न करते हैं"
                f"USER: पहले अलग-अलग लोगों के बीच हुई बातचीत का इतिहास पढ़ें, इसे लोगों के बीच छवि और संस्कृति के बारे में चर्चा के रूप में समझें; \
                    फिर वार्तालाप में वह सामग्री ढूंढें जो छवि सामग्री विवरण और संस्कृति संबंधी चर्चा से संबंधित हो; \
                    अंततः छवि सामग्री विवरण और संस्कृति संबंधी चर्चा से आपने जो सीखा है उसका सारांश प्रदान करें। "
                f"बातचीत और मानव-जैसी संवाद शैली में होना याद रखें। प्रतिक्रिया को 2 वाक्यों तक सीमित करें। अपना उत्तर अंग्रेजी में दें। \nASSISTANT:")

RO_prompt_3 = (f"SYSTEM: Sunteți o persoană din {romania_agent.role}, cunoașteți și urmați foarte bine cultura {romania_agent.role}, dar nu aveți prea multe cunoștințe despre alte culturi. \
                    Rămâneți la rolul dvs. de persoană din {romania_agent.role}. În calitate de om de la {romania_agent.role}, generați întotdeauna un limbaj conversațional în stil de dialog asemănător omului"
                f"USER: Citiți mai întâi istoricul conversațiilor dintre diferiți oameni, înțelegeți asta ca o discuție despre imagine și cultura între oameni; \
                    apoi găsiți conținutul conversației care se referă la descrierea conținutului imaginii și discuția legată de cultură; \
                    în cele din urmă, oferiți un rezumat a ceea ce ați învățat din descrierea conținutului imaginii și din discuția legată de cultură. "
                f"Nu uitați să fiți conversațional și în stilul dialogului uman. Limitați răspunsul la 2 propoziții. Trimiteți răspunsul în engleză \nASISTANT:")

CN_prompt_3 = (f"SYSTEM: 您是来自{china_agent.role}的人，您非常了解并遵循{china_agent.role}的文化，但对其他文化了解不多。\
                坚守您作为来自{china_agent.role}的人的角色。作为来自{china_agent.role}的人，您总是以类似人类的对话风格生成对话语言"
                f"USER: 首先阅读不同人之间的对话历史，将其理解为人们之间关于图像和文化的讨论；\
                然后在对话中找到与图像内容描述相关的内容，以及与文化相关的讨论；\
                最后提供您从图像内容描述和与文化相关的讨论中了解到的内容的摘要。"
                f"请用英文进行回复, 记住要以对话和类似人类的对话风格进行。将回复限制在2句话内。\nASSISTANT:")

multilingual_prompt_3 = [US_prompt_3, IN_prompt_3, RO_prompt_3, CN_prompt_3]


base_path = ''


def generate_caption(relative_path):


    image_source = os.path.join(base_path, relative_path)

    responses__ = []
    countries = ['USA', 'India', 'Romania', 'China']
    rounds = 2
    response = []
    agents = [usa_agent, india_agent, romania_agent, china_agent]
    questions = []
    # then ask a question according to the image. Output in the following format

    prompt_mod = (f"<image>\n"
                f"SYSTEM: You are a {moderator.role}, who is tasked to generate questions based on an image. "
                f"USER: Given the image, first, try to find as much as different objects in the image as you can; \
                    next, think of how can these observed objects related to different cultures; \
                    then, generate 20 different unique questions related to culture about the image to cover each unique object you observed. \
                    Remember to focus on the different aspects on the image (objects and humans alike) and create a comprehensive list of culture related questions. \
                    Also remember to be conversational and in human-like dialogue style. Answer in this format: <question1>\n<question2>... \nASSISTANT:")

    # prompt to focus on different objects and humans - generates a more comprehensive summary of the objects in images - not sure about culture-specific - need to check
    # prompt_mod = (f"<image>\n"
    #               f"SYSTEM: You are a {moderator.role}, who is tasked to generate questions based on an image. "
    #               f"USER: Given {image_source}, create a comprehensive list of 10 questions by focusing on humans and 4 different objects present in the {image_source}, ask two unique questions about each object (int total 4) present in {image_source} and two unique questions about humans present in {image_source}. Answer in this format: <question1>\n<question2>\n<question3>\n...<question9>\n<question10>\nASSISTANT:")


    resp = moderator.generate_response_llava_wImage(prompt_mod, image_source, device)
    # print("questions:", resp)
    questions.append(resp)
    responses__.append(f"{moderator.role}: {resp}\n")
    points = [elem.strip().split('\n') for elem in questions][0]

    # print(len(points))
    # print(points)


    #### Multi-agent interaction round 1 start 
    round1 = []

    for agent, question, prompt in zip(agents, points[0:4], multilingual_prompt_1):
        # prompt = (f"<image>\n"
        #         f"SYSTEM: You are a person from {agent.role}, you know the culture from {agent.role} pretty well, but you don't have too much knowledge for other cultures. You as a human from {agent.role} always generate conversational language in human-like dialogue style"
        #         f"USER: Remember you are from {agent.role}, first observe the image and think what you first see in the image; \
        #             then think of how the object you saw is related to your culture in {agent.role}; \
        #             next think of cultural related questions about this image\
        #             finally, generate human-like conversational language to describe the object you first saw in the image and how is this related to your culture in {agent.role} and also the question you would like to ask. \
        #             Remember to be conversational and in human-like dialogue style. Limit answer to two sentences. \nASSISTANT:")
                 # f"USER: Based on your culture {agent.role}, describe what you first see in the image based on its significance to your culture in {agent.role}. Limit answer to two sentences. \nASSISTANT:")
        resp = agent.generate_response_llava_wImage(prompt, image_source, device)
        response = [f"{agent.role}: {resp}"]
        for agent_mem in agents:
            agent_mem.add_memory(f"message from: {agent.role}, content: {resp}")
        # print(f"{agent.role}: {resp}\n")
        responses__.append(f"{agent.role} Agent: {resp}\n")
        ## question 
        # prompt = (f"<image>\n"
        #         f"SYSTEM: You are a person from {agent.role}, Your role is to ask the question given in {question}. Do not answer the question. "
        #         f"USER:  You will ask {question} to other agents. Do not answer the question. Respond in this format: <{question}> <\nASSISTANT:")
        # resp = agent.generate_response_llava(prompt, device)
        resp = [f"{agent.role} + {question}"]
        round1.append(resp)
        # print(f"{agent.role} Question: {resp}\n")
        responses__.append(f"{agent.role} Question: {resp}\n")
        
    # print(questions)

    round2 = []
    # print("---------------------------------Follow-up interaction based on memory---------------------------------")
    for i in range(1):
        for agent, question, prompt in zip(agents, points[4:8], multilingual_prompt_2):
            # if agent.memory_lst:
                last_memory = agent.memory_lst[-1]['content']
                # print(f'_________Last Memory for {agent.role}____________:', last_memory)
                #answers 
                # prompt = (
                #         f"SYSTEM: You are a person from {agent.role}, you know and follow the culture of {agent.role} very well, but you don't have too much knowledge of other cultures. \
                #         Stick to your role as a person from {agent.role}. You as a human from {agent.role} always generate conversational language in human-like dialogue style"
                #         f"USER: First read the dialogue in {round1}, understand it as a dialogue from other people; \
                #             then identify what questions are asked in the conversation; \
                #             next observe the image and think of the knowledge from your culture in {agent.role}; \
                #             finally answer the question you find in the dialogue based on the observation and the knowledge your culture from {agent.role}. \
                #             Remember to be conversational and in human-like dialogue style. Respond in this format: <answer1> <answer2> .... \nASSISTANT:")
                        # f"USER: Answer all questions asked in {round1} from your own perspective and based on your culture from {agent.role}. Be more human-like in your responses. Respond in this format: <answer1> <answer2> .... \nASSISTANT:")
                
                
                resp = agent.generate_response_llava(prompt,device)
                response = [f"{agent.role}: {resp}"]
                agent_mem.add_memory(f"message from: {agent.role}, content: {resp}")
                # print(f"{agent.role} Answer: {resp}\n")
                responses__.append(f"{agent.role} Answer: {resp}\n")
                
                # questions 
                resp2 = [f"{agent.role}: {question}"]
                round1.append(resp2)
                # for agent_mem in agents:
                #     agent_mem.add_memory(f"message from: {agent.role}, content: {resp}")
                # print(f"{agent.role} Question: {resp2}\n")
                responses__.append(f"{agent.role} Question: {resp2}\n")

    # print(responses__)


    # print("---------------------------------Summary of interactions---------------------------------")
    for agent, prompt in zip(agents, multilingual_prompt_3):
        # prompt = (f"SYSTEM: You are a person from {agent.role}, you know and follow the culture of {agent.role} very well, but you don't have too much knowledge of other cultures. \
        #             Stick to your role as a person from {agent.role}. You as a human from {agent.role} always generate conversational language in human-like dialogue style"
        #         f"USER: First read the conversation history among different people, understand this as a discussion about the image and the culture among people; \
        #             then find the contents in the conversation that related to the image contents description, and the culture related discussion; \
        #             finally provide a summary of what you have learned from the image contents description and the culture related discussion. "
        #         f"Remember to be conversational and in human-like dialogue style. Limit response to 2 sentences. \nASSISTANT:")
        resp = agent.generate_response_llava(prompt, device)
        # print(f"{agent.role} Summary: {resp}\n")
        responses__.append(f"{agent.role} Agent: {resp}\n")

    ## Sumarizer 
    # prompt_sum = (f"<image>\n"
    #             f"SYSTEM: You are a {moderator.role}, who is tasked to summarize answers."
    #             f"USER: Given the conversation history: {responses__[:-4]} and the image, \
    #                 first read the conversation history and understand this as a summary from each people in a discussion about the image description and the related cultures; \
    #                 then, from the conversation history, find what contents are related to the image description and cultura knowledge; \
    #                 finally, generate a comprehensive summary based on the conversation history contents and the image, to describe the contents of the image and the culture related knowledge about this picture. \
    #                 Answer in this format: <summary>. Limit response to 3 sentences. \nASSISTANT:")
    prompt_sum = (f"<image>\n"
            f"SYSTEM: You are a {moderator.role}, who is tasked to summarize answers."
            f"USER: Given the conversation history: {responses__[:-4]} and the image, \
                first read the conversation history and understand this as a summary from each people in a discussion about the image description and the related cultures; \
                next, from the conversation history, find what contents are about the image content description\
                then, from the conversation history, find what contents are about the image related cultura knowledge; \
                finally, generate a comprehensive summary based on the conversation history contents and the image: describe the content of the picture in the first sentence, and then describe the cultural knowledge related to the picture after that. \
                Answer in this format: <summary>. Limit response to 3 sentences. \nASSISTANT:")
    resp = summarizer.generate_response_llava_wImage(prompt_sum, image_source=image_source, device=device)
    # print(f"summarizer: {resp}")
    responses__.append(f"{summarizer.role}: {resp}\n")

    caption = resp

    return caption





df['llava_agent_caption'] = [generate_caption(path) for path in tqdm(df['file_path'], desc="Generating Captions")]

# Save the DataFrame with captions
output_file_path = ''
df.to_csv(output_file_path, index=False)

print("Captions generated and saved successfully.")



# output_df = pd.DataFrame({
#     "responses": responses__
# })


# pd.DataFrame(responses__).to_csv("testOutputs.csv")
