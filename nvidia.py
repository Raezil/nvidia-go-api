from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import sys
class NVIDIAService:
    def __init__(self, api_key, model):
        self.client = ChatNVIDIA(
            model=model,
            api_key=api_key, 
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        
    def run(self, prompt):
        return ''.join(chunk.content for chunk in self.client.stream([{"role": "user", "content": prompt}]))

prompt = sys.argv[1]
model = sys.argv[2]
nvidia = NVIDIAService(api_key=os.environ['NVIDIA_API'], model=model)
print(nvidia.run(prompt))