#!pip install transformers googletrans==4.0.0-rc1

import torch
from transformers import BertForQuestionAnswering, BertTokenizer, logging
from googletrans import Translator

logging.set_verbosity_error()

print("Loading BERT model...")
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
print("Model successfully loaded.\n")

translator = Translator()

def translate_text(text, src_lang=None, dest_lang='en'):
    """Translate text using Google Translate."""
    try:
        if src_lang:
            result = translator.translate(text, src=src_lang, dest=dest_lang)
        else:
            result = translator.translate(text, dest=dest_lang)
        return result.text
    except Exception as e:
        print("Translation error:", e)
        return text

def question_answer(question, context):
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx + 1
    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    if end_idx < start_idx:
        return "Unable to find the answer."

    answer = tokens[start_idx]
    for i in range(start_idx + 1, end_idx + 1):
        if tokens[i].startswith("##"):
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        return "Unable to find the answer."

    return answer

if __name__ == "__main__":
    print("=== Multilingual Question Answering with BERT ===")
    context = input("Enter your context (English or Romanian):\n")
    question = input("\nEnter your question (English or Romanian):\n")

    detected_context_lang = translator.detect(context).lang
    detected_question_lang = translator.detect(question).lang

    if detected_context_lang != 'en':
        context_en = translate_text(context, src_lang=detected_context_lang, dest_lang='en')
    else:
        context_en = context

    if detected_question_lang != 'en':
        question_en = translate_text(question, src_lang=detected_question_lang, dest_lang='en')
    else:
        question_en = question

    answer_en = question_answer(question_en, context_en)

    answer_ro = translate_text(answer_en, src_lang='en', dest_lang='ro')

    print("\nDetected languages: context =", detected_context_lang, ", question =", detected_question_lang)
    print("\nAnswer (English):", answer_en)
    print("Answer (Romanian):", answer_ro)
