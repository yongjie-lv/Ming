import torch
import os
from Ming_Uni.MingUniInference import Ming_Uni_Inference
from Ming_Uni.process import MyProcessor
device = torch.cuda.current_device()
device = torch.device(device)

model_path='../Ming-Lite-Uni/'
model = Ming_Uni_Inference(model_path)
model.to(torch.bfloat16)
model.to(device)
model.eval()

llm_model=os.path.join(model_path, 'qwen2_5_llm')
my_proc=MyProcessor(llm_model)

image_file = "tests/cake.jpg"
prompt = "add a candle on top of the cake"
inputs = my_proc.process(image_file=image_file, prompt=prompt, device=device)

result = model.image_gen_generate(inputs, steps=30, seed=42, cfg=5.0, height=512, width=512)[1]
result.save("result.png")