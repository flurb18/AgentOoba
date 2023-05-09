import os
import sys
import time
import gradio as gr
import re

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.agents import load_tools
from langchain.tools import Tool
from typing import Dict

import modules
from modules import chat, shared
from modules.text_generation import generate_reply

CTX_MAX = 16384
VERBOSE=True
MAX_TASKS_DEFAULT=6
RECURSION_DEPTH_DEFAULT=3

HUMAN_PREFIX = "Prompt:"
ASSISTANT_PREFIX = "Response:"

# Define your Langchain tools here
SEARX_HOST = "https://searxng.nicfab.eu/"
TOP_K_WIKI = 5
WOLFRAM_APP_ID = "YPXU88-L8EJUPX4L6"
# The keys here must match tool.name
TOOL_DESCRIPTIONS = {
    "Wikipedia" : "A collection of articles on various topics. Used when the task at hand is researching or acquiring general surface-level information about any topic. Input is a topic; the tool will then save general information about the topic to memory.",
    "Searx Search" : "A URL search engine. Used when the task at hand is searching the internet for websites that mention a certain topic. Input is a search query; the tool will then save URLs for popular websites that reference the search query to memory.",
    "Wolfram Alpha" : "A multipuporse calculator and information search engine. Used for mathematical computations and looking up specific numeric information. Input is a query or directive to calculate an expression; the tool will then save the expression and the result of the evaluation of that expression to memory.\nExample: Input - 'derivative of x^2' Output - 'derivative of x^2 is 2x'"
}

ENABLED_TOOLS = ["wikipedia", "searx-search", "wolfram-alpha"]
Tools = load_tools(
    ENABLED_TOOLS,
    searx_host=SEARX_HOST,
    top_k_results=TOP_K_WIKI,
    wolfram_alpha_appid=WOLFRAM_APP_ID
)

OutputCSS = """
<style>
.oobaAgentBase {
  font-size: 1rem;
}

.oobaAgentOutput {
  font-size: 1rem;  
}

.oobaAgentOutputThinking {
  font-size: 1rem;
  border-left: 2px solid orange;
}
</style>
"""

def ooba_call(prompt, stop = None):
    generator = generate_reply(prompt, shared.persistent_interface_state, stopping_strings=stop)
    answer = ''
    for a in generator:
        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]
    if VERBOSE:
        print(f"PROMPT: {prompt}\n", file=sys.stderr)
        print(f"ANSWER: {answer}\n", file=sys.stderr)
    return answer

class Objective:
    def __init__(self, objective, recursion_max, max_tasks, recursion_level, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        self.recursion_max = recursion_max
        self.max_tasks = max_tasks
        self.tasks = []
        self.done = (recursion_level == recursion_max)
        self.current_task_idx = 0
        self.output = ""
        if self.assess_model_ability():
            self.done = True
            self.output= f"MODEL OUTPUT {self.do_objective()}"
            return
        tool_found, tool, tool_input = self.assess_tools()
        if tool_found:
            self.done = True
            self.output = f"TOOL FOUND {tool.name} input={tool_input}"
        else:
            self.tasks = self.split_objective()
            if len(self.tasks) == 0:
                self.done = True

    def make_prompt(self, directive, objs):
        directive = "\n".join([line.strip() for line in (directive.split("\n") if "\n" in directive else [directive])])[:CTX_MAX]
        directive = directive.replace("_TASK_", f"Objective {self.recursion_level}").strip()
        objstr = f"Objectives:\n{self.prompt_objective_context()}\n\n" if objs else ""
        return f"{HUMAN_PREFIX}\n{objstr}Instructions:\n{directive}\n\n{ASSISTANT_PREFIX}"

    def assess_model_ability(self):
        directive = f"Assess whether or not you are capable of completing _TASK_ entirely with a single output. Remember that you are a large language model whose only tool is outputting text. If you are certain that you can achieve _TASK_ to its full extent with a single output, respond with the word 'Yes'. Otherwise, respond with the word 'No'. If you are unsure, if you need clarification, or if you think you can help complete _TASK_ partially but not fully, respond with the word 'No'."
        prompt = self.make_prompt(directive, True)
        response = ooba_call(prompt).strip()
        return 'yes' in response.lower()

    def do_objective(self):
        directive = f"It has been determined that you are able to achieve _TASK_. Complete _TASK_ to the best of your ability. Respond with your output for completing _TASK_ and nothing else."
        response = ooba_call(self.make_prompt(directive, True)).strip()
        return response
    
    def split_objective(self):
        directive = f"Develop a plan to complete _TASK_, keeping in mind why _TASK_ is desired. The plan should come as a list of tasks, each a step in the process of completing _TASK_. The list should be written in the order that the tasks must be completed. The scope of the list should be limited to that which is strictly necessary to complete _TASK_. The number of tasks in the list should be between 1 and {self.max_tasks}. Keep the descriptions of the tasks short but descriptive. Do not include any tasks that are achieved by completing a task in our objectives or in a task list for one of our objectives. Respond with the numbered list in the following format:\n1. (first task to be completed)\n2. (second task to be completed)\n3. (third task to be completed)\netc. Do not include any text in your response other than the list; do not ask for clarifications."
        prompt = self.make_prompt(directive, True)
        response = ooba_call(prompt).strip()
        list_pos = response.find("1")
        if list_pos == -1:
            return []
        response = response[list_pos:]
        return [ item[2:].strip(" -*[]()") for item in (response.split("\n") if "\n" in response else [response]) ]
    
    def assess_tools(self):
        for tool in Tools:
            directive = f"You have access to the following tool:\n\nTool name: {tool.name}\nTool description: {TOOL_DESCRIPTIONS[tool.name]}\n\nAsses whether it is possible to achieve _TASK_ by providing a single input to the tool. If you think it is not possible, respond with the word 'No'. If you are unsure, respond with the word 'No'. If you think the tool would assist in completing _TASK_ partially but wouldn't be able to complete _TASK_ entirely by itself, respond with the word 'No'. If you need clarification, respond with the word 'No'. Otherwise, if none of the previous criteria apply and if you are absolutely certain that a single usage of the tool can completely achieve _TASK_, respond with the word 'Yes'. As a reminder, your response should just be the word 'Yes' or the word 'No' and nothing else. Do not respond with anything besides 'Yes' or 'No'; do not provide any explanation or reasoning."
            prompt = self.make_prompt(directive, True)
            response = ooba_call(prompt).strip().lower()
            negative_responses = ["no","cannot", "can't", "cant"]
            if not any([neg in response for neg in negative_responses]):
                directive = f"You have access to the following tool:\n\nTool name: {tool.name}\nTool description: {TOOL_DESCRIPTIONS[tool.name]}\n\nIt has been determined that the tool is capable of achieving _TASK_ in its entirety. Create an input for the tool that would achieve _TASK_ upon being passed to the tool, and respond with the created input. If no such input is possible, respond with the word 'cannot'. Do not include anything in your response other than the created input for the tool or the word 'cannot' depending on the criteria above."
                prompt = self.make_prompt(directive, True)
                response = ooba_call(prompt).strip().lower()
                negative_responses = ["cannot", "can't", "cant"]
                if not any([neg in response for neg in negative_responses]):
                    return True, tool, response
        return False, None, None
    
    def prompt_objective_context(self):
        reverse_context = []
        p_it = self
        r = self.recursion_level
        reverse_context.append(f"")
        while p_it.parent:
            p_it = p_it.parent
            parent_task_list_str = "\n".join([f"Objective {r-1}, Task {str(i+1)}: {p_it.tasks[i] if isinstance(p_it.tasks[i], str) else p_it.tasks[i].objective}" for i in range(len(p_it.tasks))])
            reverse_context.append(f"We have developed the following numbered list of tasks that we must complete to achieve Objective {r-1}:\n{parent_task_list_str}\n\nThe current task that we are at among these is Objective {r-1}, Task {p_it.current_task_idx+1}. We will refer to Objective {r-1}, Task {p_it.current_task_idx+1} as Objective {r}.")
            r -= 1
        assert r == 1
        reverse_context.append(f"Objective 1 is what we ultimately want to achieve. Objective 1 is: {p_it.objective}")
        reverse_context.reverse()
        return "\n".join(reverse_context)
    
    def process_current_task(self):
        if self.current_task_idx == len(self.tasks):
            self.current_task_idx = 0
            self.done = all([(isinstance(task, str) or task.done) for task in self.tasks])
        if not self.done:
            current_task = self.tasks[self.current_task_idx]
            if isinstance(current_task, str):
                self.tasks[self.current_task_idx] = Objective(
                    current_task,
                    self.recursion_max,
                    self.max_tasks,
                    self.recursion_level + 1,
                    parent=self
                )
                AgentOobaVars["memory"]["processed-tasks"].append(current_task)
                self.current_task_idx += 1
                if self.current_task_idx == len(self.tasks):
                    self.current_task_idx = 0
                    if self.parent:
                        self.parent.current_task_idx += 1
            else:
                current_task.process_current_task()
        else:
            if self.parent:
                self.parent.current_task_idx += 1

    def to_string(self, select):
        out = f'OBJECTIVE: {self.objective}<ul class="oobaAgentOutput">'
        if self.output:
            out += f'<li class="oobaAgentOutput">{self.output}</li>'
        else:
            for task in self.tasks:
                if isinstance(task, str):
                    thinking = False
                    current_task_iterator = self
                    task_idx = current_task_iterator.tasks.index(task)
                    while (current_task_iterator.current_task_idx == task_idx):
                        if not current_task_iterator.parent:
                            thinking = True
                            break;
                        task_it_parent = current_task_iterator.parent
                        task_idx = task_it_parent.tasks.index(current_task_iterator)
                        current_task_iterator = task_it_parent
                    task_disp_class = "oobaAgentOutputThinking" if thinking and select else "oobaAgentOutput"
                    out += f'<li class="{task_disp_class}">{task}</li>'
                else:
                    out += f'<li class="oobaAgentOutput">{task.to_string(select)}</li>'
        out += "</ul>"
        return out

def ui():
    with gr.Column():
        with gr.Column():
            output = gr.HTML(label="Output", value="")
            user_input = gr.Textbox(label="Goal for AgentOoba") 
            max_tasks_slider = gr.Slider(
                label="Max tasks in a list",
                minimum=3,
                maximum=12,
                step=1,
                value=MAX_TASKS_DEFAULT,
                interactive=True
            )
            recursion_level_slider = gr.Slider(
                label="Recursion Depth",
                minimum=1,
                maximum=7,
                step=1,
                value=RECURSION_DEPTH_DEFAULT,
                interactive=True
            )
            with gr.Row():
                submit_button = gr.Button("Execute", variant="primary")
                cancel_button = gr.Button("Cancel")

    AgentOobaVars["submit-event-1"] = submit_button.click(
        modules.ui.gather_interface_values,
        inputs = [shared.gradio[k] for k in shared.input_elements],
        outputs = shared.gradio['interface_state'],
    ).then(mainloop, inputs=[user_input, recursion_level_slider, max_tasks_slider], outputs=output)

    AgentOobaVars["submit-event-2"] = user_input.submit(
        modules.ui.gather_interface_values,
        inputs = [shared.gradio[k] for k in shared.input_elements],
        outputs = shared.gradio['interface_state'],
    ).then(mainloop, inputs=[user_input, recursion_level_slider, max_tasks_slider], outputs=output)

    def cancel_agent():
        AgentOobaVars["main-objective"].done = True
        output.value = ""
    
    cancel_event = cancel_button.click(
        cancel_agent,
        None,
        None,
        cancels = [AgentOobaVars["submit-event-1"], AgentOobaVars["submit-event-2"]]
    )
    
AgentOobaVars = {
    "submit-event-1" : None,
    "submit-event-2" : None,
    "waiting-input" : False,
    "recursion-level" : RECURSION_DEPTH_DEFAULT,
    "max-tasks" : MAX_TASKS_DEFAULT,
    "memory" : {},
    "main-objective": None
}
            
def mainloop(ostr, r, max_t):
    AgentOobaVars["recursion-level"] = r
    AgentOobaVars["max-tasks"] = max_t
    AgentOobaVars["memory"]["processed-tasks"] = []
    yield f"{OutputCSS}<br>Thinking...<br>"
    AgentOobaVars["main-objective"] = Objective(ostr, r, max_t, 1)
    while (not AgentOobaVars["main-objective"].done):
        yield f'{OutputCSS}<div class="oobaAgentBase"><br>{AgentOobaVars["main-objective"].to_string(True)}<br>Thinking...</div>'
        AgentOobaVars["main-objective"].process_current_task()
        if AgentOobaVars["waiting-input"]:
            yield f'{OutputCSS}<div class="oobaAgentBase"><br>{AgentOobaVars["main-objective"].to_string(True)}<br>Waiting for user input</div>'
        # Give GPU a second to breathe :)
        time.sleep(2)
        while AgentOobaVars["waiting-input"]:
            time.sleep(0.1)
    yield f'{OutputCSS}<div class="oobaAgentBase"><br>{AgentOobaVars["main-objective"].to_string(False)}<br>Done!</div>'
