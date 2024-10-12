import sys
sys.path.append('/nfs/turbo/coe-mihalcea/shared_data/Long-CLIP')
import clip
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd
import os
from tqdm import tqdm
from model import longclip
import nltk
from nltk.corpus import wordnet as wn


# =========================================================== CLIP/LONGCLIP score

# Function to truncate text without breaking words and ensure a period at the end
def truncate_text(text, max_length):
    if len(text) > max_length:
        truncated = text[:max_length].rsplit(' ', 1)[0]
        if not truncated.endswith('.'):
            truncated += '.'
        return truncated
    return text if text.endswith('.') else text + '.'

# def truncate_text(text, max_length):
#     if len(text) > max_length - 2:  # Subtracting 2 for special tokens (e.g., start and end tokens)
#         truncated = text[:max_length - 2].rsplit(' ', 1)[0]
#         if not truncated.endswith('.'):
#             truncated += '.'
#         return truncated
#     return text if text.endswith('.') else text + '.'

# Functions for calculating CLIP and LongCLIP scores
def get_clip_score(image_path, text, model, preprocess, first_sentence_only=False, max_length=77):
    image = Image.open(image_path)

    if not isinstance(text, str):
        print(f"Warning: Expected text but got {type(text)} at {image_path}. Defaulting to empty string.")
        text = ""

    if first_sentence_only:
        text = text.split('.')[0].strip()

    text = truncate_text(text, max_length)
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score

def get_longclip_score(image_path, text, model, preprocess, first_sentence_only=False, max_length=248):
    image = Image.open(image_path)

    if not isinstance(text, str):
        print(f"Warning: Expected text but got {type(text)} at {image_path}. Defaulting to empty string.")
        text = ""

    if first_sentence_only:
        text = text.split('.')[0].strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = truncate_text(text, max_length)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = longclip.tokenize([text]).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    longclip_score = torch.matmul(image_features, text_features.T).item()
    return longclip_score


# =========================================================== Culture Metrics

# Function to load cultural words
def load_cultural_words(filename):
    try:
        with open(filename, 'r') as file:
            cultural_words = {line.strip().lower() for line in file}
        return cultural_words
    except IOError:
        print("Error: File does not appear to exist.")
        return set()

cultural_words = load_cultural_words('MultiAgent_Multicultural_ImageUnderstanding/test_longju/eval/culture_words.txt')

# Functions to calculate cultural number ratio and count
def calculate_cnr(caption, cultural_words):
    if not isinstance(caption, str):
        print(f"Warning: Expected text but got {type(caption)}. Defaulting to empty string.")
        caption = ""

    words = set(caption.lower().split())
    cultural_count = sum(1 for word in words if word in cultural_words)
    return cultural_count / len(words) if words else 0

def calculate_cultural_number(caption, cultural_words):
    if not isinstance(caption, str):
        print(f"Warning: Expected text but got {type(caption)}. Defaulting to empty string.")
        caption = ""

    words = set(caption.lower().split())
    cultural_count = sum(1 for word in words if word in cultural_words)
    return cultural_count if words else 0


# =========================================================== Completeness

def get_wordnet_related_words(word):
    related_words = set()
    for syn in wn.synsets(word):
        # Add the synonyms
        related_words.update(syn.lemma_names())
        # Add the hyponyms
        related_words.update(lem.name() for hypon in syn.hyponyms() for lem in hypon.lemmas())
        # Add the hypernyms
        related_words.update(lem.name() for hyper in syn.hypernyms() for lem in hyper.lemmas())
    # Normalize and remove underscores
    return set(w.replace('_', ' ').lower() for w in related_words)

def calculate_completeness(tag_string, caption_string):

    if not isinstance(caption_string, str):
        print(f"Warning: Expected text but got {type(caption_string)} at {image_path}. Defaulting to empty string.")
        caption_string = ""
    
    tags = set(tag_string.split(", "))
    expanded_tags = set()
    # Expand each tag with WordNet related words
    for tag in tags:
        expanded_tags.update(get_wordnet_related_words(tag))
    caption_words = set(caption_string.lower().split())
    # Calculate the intersection of expanded tags and caption words
    intersection = expanded_tags.intersection(caption_words)
    # Return the ratio of matched tags to total tags
    if len(tags) == 0:
        return 0
    return len(intersection) / len(tags)









# Main processing ===========================================================================
# base_path = '/nfs/turbo/coe-mihalcea/shared_data/GD-VCR/X_VCR/'
# base_path = '/nfs/turbo/coe-mihalcea/shared_data/dollarstreet'
# base_path = '/nfs/turbo/coe-mihalcea/shared_data/GeoDE/images'
base_path = '/nfs/turbo/coe-mihalcea/shared_data/cvqa/data/saved_images_RIC'

# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/caption_results/llava_agent_cot_2r_correct_GeoDE.csv'
# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/caption_results/llava_ft_specific_captions_GeoDE.csv'
# file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/caption_results/llava_agent_cot_multilingual_dollarstreet_RIC500.csv'
file_path = 'MultiAgent_Multicultural_ImageUnderstanding/test_longju/caption_results/llava_agent_cot_ablation3r_correct_CVQA.csv'

df = pd.read_csv(file_path)

clip_scores = []
longclip_scores = []
clip1_scores = []
longclip1_scores = []
cnrs = []
c_numbers = []
completeness = []

device = "cuda" if torch.cuda.is_available() else "cpu"
longclip_model, longclip_preprocess = longclip.load("/nfs/turbo/coe-mihalcea/shared_data/Long-CLIP/checkpoints/longclip-L.pt", device=device)
clip_model, clip_preprocess = clip.load('ViT-B/32')

for _, row in tqdm(df.iterrows(), total=len(df), desc="evaluating"):
    image_path = os.path.join(base_path, row['image_path'])
    caption = row['llava_agent_caption']
    tag = row['tag']
    
    # clip_scores.append(get_clip_score(image_path, caption, clip_model, clip_preprocess))
    # longclip_scores.append(get_longclip_score(image_path, caption, longclip_model, longclip_preprocess))
    # clip1_scores.append(get_clip_score(image_path, caption, clip_model, clip_preprocess, first_sentence_only=True))
    # longclip1_scores.append(get_longclip_score(image_path, caption, longclip_model, longclip_preprocess, first_sentence_only=True))
    # cnrs.append(calculate_cnr(caption, cultural_words))
    # c_numbers.append(calculate_cultural_number(caption, cultural_words))
    # completeness.append(calculate_completeness(tag, caption))

    try:
        clip_scores.append(get_clip_score(image_path, caption, clip_model, clip_preprocess))
        longclip_scores.append(get_longclip_score(image_path, caption, longclip_model, longclip_preprocess))
        clip1_scores.append(get_clip_score(image_path, caption, clip_model, clip_preprocess, first_sentence_only=True))
        longclip1_scores.append(get_longclip_score(image_path, caption, longclip_model, longclip_preprocess, first_sentence_only=True))
        cnrs.append(calculate_cnr(caption, cultural_words))
        c_numbers.append(calculate_cultural_number(caption, cultural_words))
        completeness.append(calculate_completeness(tag, caption))
    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")
        clip_scores.append(0)
        longclip_scores.append(0)
        clip1_scores.append(0)
        longclip1_scores.append(0)
        cnrs.append(0)
        c_numbers.append(0)
        completeness.append(0)

df['clip_score'] = clip_scores
df['longclip_score'] = longclip_scores
df['clip1_score'] = clip1_scores
df['longclip1_score'] = longclip1_scores
df['cnr'] = cnrs
df['c_number'] = c_numbers
df['completeness'] = completeness

df.to_csv('MultiAgent_Multicultural_ImageUnderstanding/test_longju/eval/ablation_ft_and_correct/llava_agent_cot_3r_correct_CVQA_eval.csv', index=False)
