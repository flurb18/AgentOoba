import os
import sys
import time
import gradio as gr
import re
from dotenv import load_dotenv

load_dotenv()

CTX_MAX = int(os.getenv("CTX_MAX"))
VERBOSE = "true" in os.getenv("VERBOSE").lower()
MAX_TASKS_DEFAULT = int(os.getenv("MAX_TASKS_DEFAULT"))
RECURSION_DEPTH_DEFAULT = int(os.getenv("RECURSION_DEPTH_DEFAULT"))
DISTANCE_CUTOFF_DEFAULT = float(os.getenv("DISTANCE_CUTOFF_DEFAULT"))
EXPANDED_CONTEXT_DEFAULT = "true" in os.getenv("EXPANDED_CONTEXT_DEFAULT").lower()

SEARX_HOST = os.getenv("SEARX_HOST")
TOP_K_WIKI = int(os.getenv("TOP_K_WIKI"))
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

HUMAN_PREFIX = os.getenv("HUMAN_PREFIX")
ASSISTANT_PREFIX = os.getenv("ASSISTANT_PREFIX")
ASSESS_ABILITY_DIRECTIVE = os.getenv("ASSESS_ABILITY_DIRECTIVE")
DO_OBJECTIVE_DIRECTIVE = os.getenv("DO_OBJECTIVE_DIRECTIVE")
SPLIT_OBJECTIVE_DIRECTIVE = os.getenv("SPLIT_OBJECTIVE_DIRECTIVE")
ASSESS_TOOL_DIRECTIVE = os.getenv("ASSESS_TOOL_DIRECTIVE")
USE_TOOL_DIRECTIVE = os.getenv("USE_TOOL_DIRECTIVE")

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.agents import load_tools
from langchain.tools import Tool
from typing import Dict

import chromadb
from chromadb.config import Settings

class ChromaInstance:
    def __init__(self, cutoff):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(name="processed-tasks")
        self.distance_cutoff = cutoff

    def add_tasks(self, tasks, task_ids):
        self.collection.add(
            documents = tasks,
            ids = task_ids
        )

    def task_exists(self, task):
        results = self.collection.query(
            query_texts=[task],
            n_results=1
        )
        if len(results["distances"][0]) == 0:
            return False
        return results["distances"][0][0] < self.distance_cutoff


import modules
from modules import chat, shared
from modules.text_generation import generate_reply


# Define your Langchain tools here
# The keys here must match tool.name
TOOL_DESCRIPTIONS = {
    "Wikipedia" : "A collection of articles on various topics. This tool can complete tasks such as researching or acquiring information about any topic. Input is a topic; the tool will then output general information about the topic.",
    "Searx Search" : "A URL search engine. This tool can complete tasks such as searching the internet for websites that mention a certain topic. Input is a search query; the tool will then output URLs for popular websites that reference the search query.",
    "Wolfram Alpha" : "A multipuporse calculator and information search engine. Used for mathematical computations and looking up specific numeric information. Input is a query or directive to calculate an expression; the tool will then output the expression and the result of the evaluation of that expression.\nExample: Input - 'derivative of x^2' Output - 'derivative of x^2 is 2x'"
}

ENABLED_TOOLS = ["wikipedia", "searx-search"]
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
        print(f"-----------------------INPUT-----------------------\n{prompt}\n", file=sys.stderr)
        print(f"----------------------OUTPUT-----------------------\n{answer}\n", file=sys.stderr)
    return answer

class Objective:
    def __init__(self, objective, task_idx, recursion_max, max_tasks, recursion_level, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        self.recursion_max = recursion_max
        self.max_tasks = max_tasks
        self.tasks = []
        self.done = (recursion_level == recursion_max)
        self.parent_task_idx = task_idx
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
            output_tasks = self.split_objective()
            self.tasks = [task for task in output_tasks if not AgentOobaVars["processed-task-storage"].task_exists(task)]
            if VERBOSE and len(self.tasks) < len(output_tasks):
                print("Tasks pruned\n", file=sys.stderr)
                before = "\n".join(output_tasks)
                after = "\n".join(self.tasks)
                print(f"Before:\n{before}\n", file=sys.stderr)
                print(f"After:\n{after}\n", file=sys.stderr)
            if len(self.tasks) == 0:
                self.done = True
            else:
                AgentOobaVars["processed-task-storage"].add_tasks(self.tasks, self.id_strings())

    def id_strings(self):
        return [self.id_string(i) for i in range(len(self.tasks))]
                
    def id_string(self, task_idx):
        recur_prefix = "" if not self.parent else self.parent.id_string(self.parent_task_idx)
        return f"{recur_prefix}-obj-{self.recursion_level}-task-{task_idx+1}-"

    def make_prompt(self, directive, objs, parent_tasks):
        directive = "\n".join([line.strip() for line in (directive.split("\n") if "\n" in directive else [directive])])[:CTX_MAX]
        directive = directive.replace("_TASK_", f"Objective {self.recursion_level}").strip()
        objstr = f"Objectives:\n{self.prompt_objective_context(parent_tasks)}\n\n" if objs else ""
        return f"{HUMAN_PREFIX}\n{objstr}Instructions:\n{directive}\n\n{ASSISTANT_PREFIX}"

    def assess_model_ability(self):
        directive = AgentOobaVars["assess-ability-directive"]
        prompt = self.make_prompt(directive, True, False)
        response = ooba_call(prompt).strip()
        return 'yes' in response.lower()

    def do_objective(self):
        directive = AgentOobaVars["do-objective-directive"]
        response = ooba_call(self.make_prompt(directive, True, False)).strip()
        return response
    
    def split_objective(self):
        directive = AgentOobaVars["split-objective-directive"].replace("_MAX_TASKS_", str(self.max_tasks))
        prompt = self.make_prompt(directive, True, True)
        response = ooba_call(prompt).strip()
        list_pos = response.find("1")
        if list_pos == -1:
            return []
        response = response[list_pos:]
        return [ item[2:].strip(" -*[]()") for item in (response.split("\n") if "\n" in response else [response]) ]
    
    def assess_tools(self):
        for tool in Tools:
            tool_str = f"Tool name: {tool.name}\nTool description: {tool.description}"
            directive = AgentOobaVars["assess-tool-directive"].replace("_TOOL_", tool_str)
            prompt = self.make_prompt(directive, True, False)
            response = ooba_call(prompt).strip().lower()
            negative_responses = ["no","cannot", "can't", "cant"]
            if not any([neg in response for neg in negative_responses]):
                directive = AgentOobaVars["use-tool-directive"].replace("_TOOL_", tool_str)
                prompt = self.make_prompt(directive, True, False)
                response = ooba_call(prompt).strip()
                negative_responses = ["i cannot"]
                if not any([neg in response.lower() for neg in negative_responses]):
                    return True, tool, response
        return False, None, None
    
    def prompt_objective_context(self, include_parent_tasks):
        reverse_context = []
        p_it = self
        r = self.recursion_level
        if include_parent_tasks and self.parent:
            task_list_str = "\n".join([(task if isinstance(task, str) else task.objective) for task in self.parent.tasks])
            reverse_context.append(f"The following is a list of tasks that have already been processed:\n{task_list_str}")
        reverse_context.append(f"Objective {r} is the current task we are working on.")
        while p_it.parent:
            child = p_it
            p_it = p_it.parent
            if AgentOobaVars["expanded-context"]:
                parent_task_list_str = "\n".join([f"Objective {r-1}, Task {str(i+1)}: {p_it.tasks[i] if isinstance(p_it.tasks[i], str) else p_it.tasks[i].objective}" for i in range(len(p_it.tasks))])
                reverse_context.append(f"We have developed the following numbered list of tasks that we must complete to achieve Objective {r-1}:\n{parent_task_list_str}\n\nThe current task that we are at among these is Objective {r-1}, Task {p_it.current_task_idx+1}. We will refer to Objective {r-1}, Task {p_it.current_task_idx+1} as Objective {r}.")
            else:
                reverse_context.append(f"In order to achieve Objective {r-1}, we must complete Objective {r}. Objective {r} is: {child.objective}")
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
                    self.current_task_idx,
                    self.recursion_max,
                    self.max_tasks,
                    self.recursion_level + 1,
                    parent=self
                )
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
            with gr.Row():
                with gr.Column():
                    recursion_level_slider = gr.Slider(
                        label="Recursion Depth",
                        minimum=1,
                        maximum=7,
                        step=1,
                        value=RECURSION_DEPTH_DEFAULT,
                        interactive=True
                    )
                    distance_cutoff_slider = gr.Slider(
                        label = "Task Similarity Cutoff",
                        minimum = 0,
                        maximum = 1,
                        step = 0.01,
                        value = DISTANCE_CUTOFF_DEFAULT
                    )
                    max_tasks_slider = gr.Slider(
                        label="Max tasks in a list",
                        minimum=3,
                        maximum=12,
                        step=1,
                        value=MAX_TASKS_DEFAULT,
                        interactive=True
                    )
                expanded_context_toggle = gr.Checkbox(label="Expanded Context (runs out of memroy at high recursion)", value = EXPANDED_CONTEXT_DEFAULT)
            with gr.Row():
                submit_button = gr.Button("Execute", variant="primary")
                cancel_button = gr.Button("Cancel")
            with gr.Accordion(label="Prompting", open = False):
                aadinput = gr.TextArea(label="Assess ability directive", value = ASSESS_ABILITY_DIRECTIVE)
                dodinput = gr.TextArea(label="Do objective directive", value = DO_OBJECTIVE_DIRECTIVE)
                sodinput = gr.TextArea(label="Split objective directive", value = SPLIT_OBJECTIVE_DIRECTIVE)
                atdinput = gr.TextArea(label="Assess tool directive", value = ASSESS_TOOL_DIRECTIVE)
                utdinput = gr.TextArea(label="Use tool directive", value = USE_TOOL_DIRECTIVE)
                aaddef = gr.Textbox(visible=False, value = ASSESS_ABILITY_DIRECTIVE)
                doddef = gr.Textbox(visible=False, value = DO_OBJECTIVE_DIRECTIVE)
                soddef = gr.Textbox(visible=False, value = SPLIT_OBJECTIVE_DIRECTIVE)
                atddef = gr.Textbox(visible=False, value = ASSESS_TOOL_DIRECTIVE)
                utddef = gr.Textbox(visible=False, value = USE_TOOL_DIRECTIVE)
                reset_prompts_button = gr.Button("Reset prompts to default")

    mainloop_inputs = [
        user_input,
        recursion_level_slider,
        max_tasks_slider,
        distance_cutoff_slider,
        expanded_context_toggle,
        aadinput,
        dodinput,
        sodinput,
        atdinput,
        utdinput
    ]

    AgentOobaVars["submit-event-1"] = submit_button.click(
        modules.ui.gather_interface_values,
        inputs = [shared.gradio[k] for k in shared.input_elements],
        outputs = shared.gradio['interface_state']
    ).then(mainloop, inputs=mainloop_inputs, outputs=output)

    AgentOobaVars["submit-event-2"] = user_input.submit(
        modules.ui.gather_interface_values,
        inputs = [shared.gradio[k] for k in shared.input_elements],
        outputs = shared.gradio['interface_state']
    ).then(mainloop, inputs=mainloop_inputs, outputs=output)

    def cancel_agent():
        AgentOobaVars["main-objective"].done = True
        output.value = ""
    
    cancel_event = cancel_button.click(
        cancel_agent,
        None,
        None,
        cancels = [AgentOobaVars["submit-event-1"], AgentOobaVars["submit-event-2"]]
    )

    def reset_prompts():
        aadinput.value = ASSESS_ABILITY_DIRECTIVE
        dodinput.value = DO_OBJECTIVE_DIRECTIVE
        sodinput.value = SPLIT_OBJECTIVE_DIRECTIVE
        atdinput.value = ASSESS_TOOL_DIRECTIVE
        utdinput.value = USE_TOOL_DIRECTIVE
        
    reset_event = reset_prompts_button.click(
        lambda a,b,c,d,e: [a,b,c,d,e],
        inputs = [aaddef, doddef, soddef, atddef, utddef],
        outputs = [aadinput, dodinput, sodinput, atdinput, utdinput]
    )
    
AgentOobaVars = {
    "submit-event-1" : None,
    "submit-event-2" : None,
    "waiting-input" : False,
    "recursion-level" : RECURSION_DEPTH_DEFAULT,
    "max-tasks" : MAX_TASKS_DEFAULT,
    "expanded-context" : EXPANDED_CONTEXT_DEFAULT,
    "processed-task-storage" : None,
    "main-objective": None,
    "assess-ability-directive" : ASSESS_ABILITY_DIRECTIVE,
    "do-objective-directive" : DO_OBJECTIVE_DIRECTIVE,
    "split-objective-directive" : SPLIT_OBJECTIVE_DIRECTIVE,
    "assess-tool-directive" : ASSESS_TOOL_DIRECTIVE,
    "use-tool-directive" : USE_TOOL_DIRECTIVE
}
            
def mainloop(ostr, r, max_t, c, expanded_context, aad, dod, sod, atd, utd):
    AgentOobaVars["assess-ability-directive"] = aad
    AgentOobaVars["do-objective-directive"] = dod
    AgentOobaVars["split-objective-directive"] = sod
    AgentOobaVars["assess-tool-directive"] = atd
    AgentOobaVars["use-tool-directive"] = utd
    AgentOobaVars["recursion-level"] = r
    AgentOobaVars["max-tasks"] = max_t
    AgentOobaVars["expanded-context"] = expanded_context
    AgentOobaVars["processed-task-storage"] = ChromaInstance(c)
    AgentOobaVars["processed-task-storage"].add_tasks([ostr],["MAIN OBJECTIVE"])
    yield f"{OutputCSS}<br>Thinking...<br>"
    AgentOobaVars["main-objective"] = Objective(ostr, -1, r, max_t, 1)
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
