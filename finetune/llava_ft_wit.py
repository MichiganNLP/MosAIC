import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
import PIL.Image
import numpy as np

# bash-4.4$ python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

# bash-4.4$ pip install huggingface_hub
model_id = "llava-hf/llava-1.5-13b-hf"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

PIL.Image.MAX_IMAGE_PIXELS = 933120000


model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)


# LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""


tokenizer = AutoTokenizer.from_pretrained(model_id)


# Define special tokens
special_tokens_dict = {
    'pad_token': '[PAD]',
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
}

# Add image token as an additional special token
special_tokens_dict['additional_special_tokens'] = ['<image>']

# Add the special tokens
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Set the chat template
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE

# Update the model's embedding layer
model.resize_token_embeddings(len(tokenizer))

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            try:
                messages = example["json_ans"]
                # Add the image token to the text
                text = "<image> " + self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
                image = example["image"].resize((224, 224))
                image = np.array(image)
                if image.ndim == 2:  # Grayscale image
                    image = np.stack((image,) * 3, axis=-1)  # Convert to RGB
                elif image.ndim == 3 and image.shape[2] == 4:  # RGBA image
                    image = image[:, :, :3]  # Remove alpha channel
                images.append(image)
            except (UnidentifiedImageError, AttributeError) as e:
                print(f"Skipping image due to error: {e}")
                continue

        # Process images and texts together
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
    

data_collator = LLavaDataCollator(processor)


raw_datasets = load_dataset("")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

training_args = TrainingArguments(
    output_dir="",
    # report_to="tensorboard",
    learning_rate=1.4e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=5,
    num_train_epochs=1,
    push_to_hub=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=True,
    bf16=False
)


lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear"
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",  # need a dummy field
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_kwargs={"skip_prepare_dataset": True},
)


trainer.train()

trainer.push_to_hub()