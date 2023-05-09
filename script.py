import os
import time
import gradio as gr
import modules
from modules import chat, shared
from modules.text_generation import generate_reply

CTX_MAX = 1500
VERBOSE=True
MAX_TASKS_DEFAULT=5
RECURSION_DEPTH_DEFAULT=3
HUMAN_PREFIX="### Human: "
ASSISTANT_PREFIX="### Assistant: "

def preface_prompt(prompt: str) -> str:
    return HUMAN_PREFIX + prompt

def fix_prompt(prompt: str) -> str:
    return preface_prompt("\n".join([line.strip() for line in (prompt.split("\n") if "\n" in prompt else [prompt])])[:CTX_MAX] + "\nResponse:\n")

def fix_response(response: str) -> str:
    if (ASSISTANT_PREFIX in response):
        return response.split(ASSISTANT_PREFIX)[1]
    else:
        return response

def ooba_call(prompt: str):
    fixed_prompt = fix_prompt(prompt)
    generator = generate_reply(fixed_prompt, shared.persistent_interface_state, stopping_strings=[HUMAN_PREFIX])
    answer = ''
    for a in generator:
        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]
    answer = fix_response(answer)
    if VERBOSE:
        print(f"PROMPT: {fixed_prompt}\n")
        print(f"ANSWER: {answer}\n")
    return answer

class Objective:
    def __init__(self, objective, recursion_max, max_tasks, recursion_level, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        self.recursion_max = recursion_max
        self.max_tasks = max_tasks
        parent_iterator = self
        reverse_prompt = []
        while parent_iterator.parent:
            reverse_prompt.append(f"In order to achieve Objective Number {recursion_level-1}, we need to achieve Objective Number {recursion_level}. Objective Number {recursion_level} is: {parent_iterator.objective}")
            parent_iterator = parent_iterator.parent
            recursion_level -= 1
        assert recursion_level == 1
        reverse_prompt.append(f"Objective Number 1 is what we ultimately want to achieve. Objective number 1 is : {parent_iterator.objective}")
        reverse_prompt.reverse()
        prompt_context = "\n".join(reverse_prompt)
        prompt=f"{prompt_context}\nDevelop a list of tasks that one must complete to attain Objective Number {self.recursion_level}, keeping in mind why Objective Number {self.recursion_level} is desired. The list should be written in the order that the tasks must be completed. The list should have at most {self.max_tasks} items. Do not include items that are not directly necessary for completing Objective Number {self.recursion_level}.  Respond only with the numbered list in the following format:\n1. (first task to be completed)\n2. (second task to be completed)\n3. (third task to be completed)\netc."
        response = ooba_call(prompt)
        self.tasks = [ item[2:].strip() for item in (response.split("\n") if "\n" in response else [response]) ]
        if len(self.tasks) == 0:
            print("Empty response from model")
            self.done = True
        else:
            self.current_task_idx = 0
            self.done = False
            
    def process_current_task(self):
        current_task = self.tasks[self.current_task_idx]
        if isinstance(current_task, str):
            if self.recursion_level < self.recursion_max:
                o = Objective(current_task, self.recursion_max, self.max_tasks, self.recursion_level + 1, parent=self)
                self.tasks[self.current_task_idx] = o
            self.current_task_idx += 1
        else:
            current_task.process_current_task()
            self.current_task_idx += 1
        if self.recursion_level == self.recursion_max:
            self.done = True
        if self.current_task_idx == len(self.tasks):
            self.current_task_idx = 0
            self.done = all([task.done for task in self.tasks])

    def to_string(self, indent):
        output = f'OBJECTIVE: {self.objective}<br><ul style="font-size:1rem">'
        for task in self.tasks:
            if isinstance(task, str):
                output += f'<li style="font-size:1rem">{task}</li>'
            else:
                output += f'<li style="font-size:1rem">{task.to_string(0)}</li>'
        output += "</ul>"
        return output

def ui():
    with gr.Column():
        with gr.Column():
            user_input = gr.Textbox(label="Goal for AgentOoba")
            output = gr.HTML(label="Output", value="")
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

            submit_event = submit_button.click(
                modules.ui.gather_interface_values,
                inputs = [shared.gradio[k] for k in shared.input_elements],
                outputs = shared.gradio['interface_state']
            ).then(mainloop, inputs=[user_input, recursion_level_slider, max_tasks_slider], outputs=output)

            submit_event_2 = user_input.submit(
                modules.ui.gather_interface_values,
                inputs = [shared.gradio[k] for k in shared.input_elements],
                outputs = shared.gradio['interface_state']
            ).then(mainloop, inputs=[user_input, recursion_level_slider, max_tasks_slider], outputs=output)

            def doNothing():
                pass
            cancel_event = cancel_button.click(doNothing, None, None, cancels=[submit_event, submit_event_2])
            
def mainloop(ostr, r, max_t):
    yield "<br>Thinking...<br>"
    o = Objective(ostr, r, max_t, 1)
    while (not o.done):
        yield f"<br>MASTER PLAN:<br>{o.to_string(0)}<br>Thinking..."
        o.process_current_task()
        # Give GPU a second to breathe :)
        time.sleep(2)
    yield f"<br>MASTER PLAN:<br>{o.to_string(0)}<br>Done!"
