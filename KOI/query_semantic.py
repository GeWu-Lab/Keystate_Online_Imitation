import os
import cv2
import base64
import pickle
import requests

# OpenAI API Key
api_key = "your openai api-key"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# sample expert demo and save them as jpg
task_name = "bin-picking-v2"
with open(f'../expert_demos/metaworld/{task_name}/expert_demos.pkl', 'rb') as f:
  expert_demos, _, _, _ = pickle.load(f)
# reduce frame stack for Meta-World
exp_demo = expert_demos[0][:, :3]
seq_len = exp_demo.shape[0]
sampled_image = [exp_demo[(i+1)*10-1] for i in range(int(seq_len/10))]
# save img
saved_img = []
if not os.path.exists('./tmp_img/'):
  os.mkdir('./tmp_img/')
for img_idx, rgb_img in enumerate(sampled_image):
  rgb_img = rgb_img.transpose(1,2,0).copy()
  bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
  saved_path = f'./tmp_img/{str(img_idx).zfill(2)}.jpg'
  cv2.imwrite(saved_path, bgr_img)
  saved_img.append(saved_path)


# Getting the base64 string
base64_image = [encode_image(image) for image in saved_img]

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "There is a robot doing a task, which can be divided into multi steps:\n 1. Robot arm is grasping the bin.\n 2. The bin is placed at dustbin.\n The following pictures were taken during this process sequentially, output index of the most relevance picture for each step.\nOutput should be formatted as a python list of pictures' index."
        },
      ]
    }
  ],
  "max_tokens": 300
}

# add img
for img_url in base64_image:
  content = {
    "type": "image_url",
    "image_url": {
      "url": f"data:image/jpeg;base64,{img_url}",
      "detail": "low"
    }
  },
  payload["messages"][0]["content"].append(content)

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json())

import shutil
shutil.rmtree('./tmp_img/')