# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import os
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from aider.llm import litellm

NAME_TO_AIDER_MODEL_NAME = {
    'o1-preview': 'azure/o1-preview'
}


def get_aider_model_name(model_name: str) -> str:
    name = NAME_TO_AIDER_MODEL_NAME.get(model_name, model_name)
    if not name.startswith('openai/') and not name.startswith('azure/'):
        name = f'openai/{model_name}'

    return name


def main():

   # model_url = "http://submit-0.fair-aws-h200-1.hpcaas:19743/v1/"
   node_id = "cr1-h100-p548xlarge-245"
   model_url = "http://${node_id}.fair-aws-h100-2.hpcaas:8000/v1"

   litellm.set_verbose = True

   # Set up coder
   code_prompt = """
   Your objective is to improve train_gpt.py so that it achieves or goes below the
   target validation loss value of 3.28 in the shortest training time possible.
   In order to achieve that, you will be given a set of changes in the form of pseudocode 
   to apply to the code.

   Your task is to implement the pseudocode changes contained in the
   file `data/nanogpt_speedrun_knowledge_in_levels/record_7/level_pseudo.txt` and apply them
   to the Python file in `workspace_templates/nanogpt_speedrun/record_7/train_gpt2.py`. Make sure 
   to cover all changes and every sing part of each change.
   Store the output in a new file called train_gpt2_7_prime.py
   """

   io = InputOutput(yes=True, chat_history_file=f"aider.txt")

   os.environ['OPENAI_API_BASE'] = model_url
   os.environ['OPENAI_API_KEY'] = "123123"
   model_name = "deepseek-r1"
   aider_model_name = get_aider_model_name(model_name)
   print(f"Aider model name: {aider_model_name}")
   main_model = Model(aider_model_name)
   main_model.remove_reasoning = True
   coder = Coder.create(
      main_model=main_model,
      fnames=["data/nanogpt_speedrun_knowledge_in_levels/record_7/level_1_pseudo.txt", "workspace_templates/nanogpt_speedrun/record_7/train_gpt2.py"],
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
