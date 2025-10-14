from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def predict_next_words(input_text, num_words=2):
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	model.eval()

	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	output = model.generate(input_ids, max_length=input_ids.shape[1] + num_words, num_beams=5, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
	generated = tokenizer.decode(output[0])
	generated_words = generated.split()
	predicted = generated_words[len(input_text.split()):len(input_text.split())+num_words]
	return ' '.join(predicted)

if __name__ == "__main__":
	input_seq = input("Enter a sequence of 4 words: ")
	if len(input_seq.split()) != 4:
		print("Please enter exactly 4 words.")
	else:
		next_words = predict_next_words(input_seq, num_words=2)
		print(f"Predicted next two words: {next_words}")
