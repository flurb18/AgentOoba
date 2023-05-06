import os
import requests
import gradio as gr
import modules
from modules import chat, shared
from modules.text_generation import generate_reply

CTX_MAX = 16384
DFS=False
RECURSION_LEVEL=2
MAX_TASKS=5
OBJECTIVE="Eat a tasty snack."

def fix_prompt(prompt: str) -> str:
    return "\n".join([line.strip() for line in (prompt.split("\n") if "\n" in prompt else [prompt])])[:CTX_MAX] + "\nResponse:\n"

def ooba_call(prompt: str):
    generator = generate_reply(fix_prompt(prompt), shared.persistent_interface_state, stopping_strings=[])
    answer = ''
    for a in generator:
        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]
    return answer
 
def strip_numbered_list(nl):
    result_list = []
    filter_chars = ['#', '(', ')', '[', ']', '.', ':', ' ']
    for line in nl:
        line = line.strip()
        if len(line) > 0:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                left_part = ''.join(x for x in parts[0] if not x in filter_chars)
                if left_part.isnumeric():
                    result_list.append(parts[1].strip())
                else:
                    result_list.append(line)
            else:
                result_list.append(line)
    # filter result_list
    result_list = [line for line in result_list if len(line) > 3]
    # remove duplicates
    result_list = list(set(result_list))
    return result_list

def check_relevance(task, objective):
    prompt = f"We have an objective and a task. Determine if the task is relevant to completing the objective. If it is, respond with the word 'true'. Otherwise, respond with the word 'false'. Do not provide any reasoning or clarification. The task is: {task}\nThe objective is: {objective}"
    response = ooba_call(prompt)
    return "true" in response.lower()

class Objective:
    def __init__(self, objective, recursion_level=RECURSION_LEVEL, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        prompt_context = f"The current objective is: {self.objective}\n"
        if self.parent:
            prompt_context += f"This is the current objective because it will help complete the ultimate objective, which is: {OBJECTIVE}\n"
        prompt=f"{prompt_context}\nDevelop a list of tasks that one must complete to attain the current objective. The list should have at most {MAX_TASKS} items. Respond only with the numbered list, in the order that one must complete the tasks, with each task on a new line. Don't say anything besides the list."
        response = ooba_call(prompt)
        self.tasks = strip_numbered_list(response.split("\n") if "\n" in response else [response])
        if len(self.tasks) == 0:
            print("Empty response from model")
            self.done = True
        else:
            self.current_task_idx = 0
            self.done = False
            
    def process_current_task(self):
        current_task = self.tasks[self.current_task_idx]
        if isinstance(current_task, str):
            if self.recursion_level != 0:
                o = Objective(current_task,
                              recursion_level=self.recursion_level - 1,
                              parent=self)
                self.tasks[self.current_task_idx] = o
            if not DFS:
                self.current_task_idx += 1
        else:
            current_task.process_current_task()
            if not DFS or current_task.done:
                self.current_task_idx += 1
        if self.recursion_level == 0:
            self.done = True
        if self.current_task_idx == len(self.tasks):
            if DFS:
                self.done = True
            else:
                self.current_task_idx = 0
                self.done = all([task.done for task in self.tasks])

    def to_string(self, indent):
        idt_string = "---"*indent
        output = f"{idt_string}Objective: {self.objective}\n"
        for task in self.tasks:
            if isinstance(task, str):
                output += f"---{idt_string}{task}\n"
            else:
                output += task.to_string(indent+1)
        return output

def ui():
    global DFS, RECURSION_LEVEL, MAX_TASKS, OBJECTIVE
    with gr.Column():
        with gr.Column():
            output = gr.Textbox(label="Output", elem_classes="textbox", lines=20, max_lines=100, interactive=False)
            user_input = gr.Textbox(label="Goal for AgentOoba")
            max_tasks_slider = gr.Slider(
                label="Max tasks in a list",
                minimum=2,
                maximum=15,
                step=1,
                value=MAX_TASKS,
                interactive=True
            )
            with gr.Row():
                recursion_level_slider = gr.Slider(
                    label="Recursion Depth",
                    minimum=1,
                    maximum=7,
                    step=1,
                    value=RECURSION_LEVEL,
                    interactive=True
                )
                dfs_toggle = gr.Checkbox(label="Depth-First Search", value=DFS)

            def submit(dfs,recursion_level,max_tasks):
                DFS = dfs
                RECURSION_LEVEL = recursion_level
                MAX_TASKS = max_tasks
            
            with gr.Row():
                submit_button = gr.Button("Execute", variant="primary")
                cancel_button = gr.Button("Cancel")

            submit_event = submit_button.click(
                modules.ui.gather_interface_values,
                inputs = [shared.gradio[k] for k in shared.input_elements],
                outputs = shared.gradio['interface_state']
            ).then(
                    submit, inputs=[dfs_toggle, recursion_level_slider, max_tasks_slider]
            ).then(
                    mainloop, inputs=user_input, outputs=output
            )

            def doNothing():
                pass

            cancel_event = cancel_button.click(doNothing, None, None, cancels=[submit_event])
    
def mainloop(ostr):
    yield "Thinking...\n"
    o = Objective(ostr)
    while (not o.done):
        o.process_current_task()
        yield f"MASTER PLAN:\n{o.to_string(0)}\nThinking..."
    yield f"MASTER PLAN:\n{o.to_string(0)}\nDone!"
