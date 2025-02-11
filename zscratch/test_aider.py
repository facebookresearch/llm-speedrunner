import sys

import os
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from aider.llm import litellm


def main():
   if len(sys.argv) != 2:
       print("Usage: python test_aider.py <vllm server node_id>")
       sys.exit(1)
   node_id = sys.argv[1]

   model_url = f"http://{node_id}.fair-aws-h100-2.hpcaas:8000/v1"

   litellm.set_verbose = True

   # Set up coder
   code_prompt = """
   Your task is to modify two files: "print_color.py" and "main.py".

   1. For "print_color.py":
      - Define a function named print_color(text: str, hex_color: str) that prints the given text in the color specified by the hex code.
      - You can assume that the terminal supports ANSI escape codes for color. (For example, convert the hex color into an ANSI escape sequence.)

   2. For "main.py":
      - Write code that prints the letters of the alphabet (A to Z) on the same line.
      - Each letter should be printed in a different color. Use the print_color function from print_color.py to print each letter.
      - You can choose any distinct hex color for each letter.
   """

   io = InputOutput(yes=True, chat_history_file=f"aider.txt")

   os.environ['OPENAI_API_BASE'] = model_url
   os.environ['OPENAI_API_KEY'] = "123123"
   main_model = Model("openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
   main_model.remove_reasoning = True
   coder = Coder.create(
      main_model=main_model,
      fnames=["main.py", "print_color.py"],
      io=io,
      stream=True,
      use_git=False,
      edit_format="diff",
      summarize_from_coder=False
   )

   coder_out = coder.run(code_prompt)

   if coder.summarizer_thread:
      coder.summarizer_thread.join()

   print("Coder result:")
   print(coder_out)


if __name__ == '__main__':
   main()