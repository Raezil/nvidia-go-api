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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python nvidia.py <prompt> <model>")
        sys.exit(1)
    prompt = sys.argv[1]
    model = sys.argv[2]
    nvidia = NVIDIAService(api_key=os.environ['NVIDIA_API'], model=model)
    final_answer = nvidia.tree_of_thoughts(prompt)
    print(nvidia.run(prompt))