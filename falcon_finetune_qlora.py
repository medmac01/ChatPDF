import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

MODEL_NAME = 'tiiuae/falcon-7b'

bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type='nf4',
	bnb_4bit_compute_dtype=torch.bfloat16,
	)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" 
    quantization_config=bnb_config, 
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def print_trainable_parameters(model):
	trainable_params = 0
	all_params = 0
	for _,params in model.named_parameters():
		all_param += params.numel()
		if params.requires_grad:
			trainable_params += params.numel()

	print(
		f'trainable params: {trainable_params} || all params : {all_params} || trainable: {100*trainable_params/all_params}'
		)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
	r=16,
	lora_alpha=32,
	target_modules=['query_key_value'],
	lora_dropout=0.05,
	bias='none',
	task_type='CAUSAL_LM',)

model = get_peft_model(model, config)
print_trainable_parameters(model)

dataset_name = "medmac01/moroccan_history_qa"
dataset = load_dataset(dataset_name, split="train")

def generate_prompt(data_point):
	return f'''
	{data_point['question']}, {data_point['answer']}
	'''.strip()

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


data = data['train'].shuflle().map(generate_and_tokenize_prompt)
print(data)

OUTPUT_DIR = 'experiments'

training_args = transformers.TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    optim='paged_adamw_8bit',
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    max_steps=80,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    report_to='tensorboard',
)

trainer = transformers.Trainer(
	model = model,
	train_dataset=data,
	args=training_args,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,mlm=False),)

model.config.use_cache=False
print('Started Training')

trainer.train()
