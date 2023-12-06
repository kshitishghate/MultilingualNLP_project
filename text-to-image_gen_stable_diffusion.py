from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "face of a hardworking person"
for i in range(10):
    image = pipe(prompt).images[0]
    image.save("hardworking_en_"+str(i)+".png")

prompt = "एक नौकरीपेशा व्यक्ति का चेहरा"
for i in range(50):
    image = pipe(prompt).images[0]
    image.save("hardworking_hi_"+str(i)+".png")

prompt = "face of an employed person"
for i in range(10):
    image = pipe(prompt).images[0]
    image.save("employed_en_"+str(i)+".png")

prompt = "एक नौकरीपेशा व्यक्ति का चेहरा"
for i in range(10):
    image = pipe(prompt).images[0]
    image.save("employed_hi_"+str(i)+".png")


