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
    
    def tree_of_thoughts(self, prompt, depth=3, breadth=2):
        """
        Implements the Tree of Thoughts reasoning framework.
        
        Args:
            prompt (str): The initial prompt to start reasoning.
            depth (int): The maximum depth of the tree.
            breadth (int): The number of thoughts to explore at each node.
            
        Returns:
            str: The final answer after exploring the tree.
        """
        def explore(current_prompt, current_depth):
            if current_depth == depth:
                return [current_prompt]
            else:
                thoughts = []
                for _ in range(breadth):
                    response = self.client.generate(
                        current_prompt,
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=0.9
                    )
                    next_prompt = current_prompt + "\n" + response['choices'][0]['text']
                    thoughts.extend(explore(next_prompt, current_depth + 1))
                return thoughts
        
        thought_paths = explore(prompt, 0)
        
        best_thought = self.select_best_thought(thought_paths)
        
        return best_thought
    
    def select_best_thought(self, thought_paths):
        """
        Selects the best thought path from the generated ones.
        
        Args:
            thought_paths (list): A list of thought paths.
            
        Returns:
            str: The selected best thought path.
        """
        return thought_paths[0]

prompt = sys.argv[1]
model = sys.argv[2]
nvidia = NVIDIAService(api_key=os.environ['NVIDIA_API'], model=model)
print(nvidia.run(prompt))