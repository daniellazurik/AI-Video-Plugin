from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




def analyze_transcript(transcript,model,tokenizer,device):
    prompt = f"Identify 3 interesting clips from this Hebrew podcast transcript. Output start and end times in JSON format. \n\nTranscript:\n{transcript}"

    messages = [
        {"role": "system",
         "content": "You are a helpful assistant that identifies interesting podcast segments."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


