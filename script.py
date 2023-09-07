import os
import sys
import time
import gradio as gr
import json
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.agents import load_tools
from langchain.tools import Tool

from modules import chat, shared
from modules.text_generation import generate_reply
from modules.ui import gather_interface_values, list_interface_input_elements
from modules.utils import gradio

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
GENERATE_THOUGHTS_DIRECTIVE = os.getenv("GENERATE_THOUGHTS_DIRECTIVE")
PRIMARY_DIRECTIVE = os.getenv("PRIMARY_DIRECTIVE")
SUMMARIZE_DIRECTIVE = os.getenv("SUMMARIZE_DIRECTIVE")

AgentOobaVars = {
    "verbose" : VERBOSE,
    "max-context" : CTX_MAX,
    "waiting-input" : False,
    "recursion-max" : RECURSION_DEPTH_DEFAULT,
    "max-tasks" : MAX_TASKS_DEFAULT,
    "max-summaries" : 5,
    "expanded-context" : EXPANDED_CONTEXT_DEFAULT,
    "chroma-cutoff" : DISTANCE_CUTOFF_DEFAULT,
    "processed-task-storage" : None,
    "main-objective": None,
    "tools" : {},
    "directives" : {
        "Primary directive" : PRIMARY_DIRECTIVE,
        "Assess ability directive" : ASSESS_ABILITY_DIRECTIVE,
        "Do objective directive" : DO_OBJECTIVE_DIRECTIVE,
        "Split objective directive" : SPLIT_OBJECTIVE_DIRECTIVE,
        "Assess tool directive" : ASSESS_TOOL_DIRECTIVE,
        "Use tool directive" : USE_TOOL_DIRECTIVE,
        "Generate thoughts directive" : GENERATE_THOUGHTS_DIRECTIVE,
        "Summarize directive" : SUMMARIZE_DIRECTIVE
    },
    "human-prefix" : HUMAN_PREFIX,
    "assistant-prefix" : ASSISTANT_PREFIX
}

params = {
    "display_name" : "AgentOoba",
    "is_tab" : True
}

def ooba_call(prompt, state):
    stops = [AgentOobaVars["human-prefix"], '</s>']
    generator = generate_reply(prompt, state, stopping_strings=stops)
    answer = ''
    for a in generator:
        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]
    for stop in stops:
        if stop in answer:
            answer = answer[:answer.find(stop)]
    if VERBOSE:
        print(f"-----------------------INPUT-----------------------\n{prompt}\n", file=sys.stderr)
        print(f"----------------------OUTPUT-----------------------\n{answer}\n", file=sys.stderr)
    return answer

from extensions.AgentOoba.objective import Objective

class ChromaTaskStorage:
    def __init__(self):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(name="processed-tasks")

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
        return results["distances"][0][0] < AgentOobaVars["chroma-cutoff"]

# Tools can be (hopefully, not all tested) any from https://python.langchain.com/en/latest/modules/agents/tools/getting_started.html
KNOWN_TOOLS = ["wikipedia", "searx-search", "requests_get", "requests_post"]

# Define Custom tool descriptions here. The keys here must match tool.name
CUSTOM_TOOL_DESCRIPTIONS_EXAMPLE = {
    "Wikipedia" : "A collection of articles on various topics. Used when the task at hand is researching or acquiring general surface-level information about any topic. Input is a topic; the tool will then save general information about the topic to memory.",
    "Searx Search" : "A URL search engine. Used when the task at hand is searching the internet for websites that mention a certain topic. Input is a search query; the tool will then save URLs for popular websites that reference the search query to memory.",
    "Wolfram Alpha" : "A multipuporse calculator and information search engine. Used for mathematical computations and looking up specific numeric information. Input is a query or directive to calculate an expression; the tool will then save the expression and the result of the evaluation of that expression to memory.\nExample: Input - 'derivative of x^2' Output - 'derivative of x^2 is 2x'"
}
CUSTOM_TOOL_DESCRIPTIONS = {}

Tools = load_tools(
    KNOWN_TOOLS,
    searx_host=SEARX_HOST,
    top_k_results=TOP_K_WIKI,
    wolfram_alpha_appid=WOLFRAM_APP_ID
)

def setup_tools():
    for tool in Tools:
        AgentOobaVars["tools"][tool.name] = {}
        AgentOobaVars["tools"][tool.name]["active"] = False
        AgentOobaVars["tools"][tool.name]["execute"] = False
        if tool.name in CUSTOM_TOOL_DESCRIPTIONS:
            AgentOobaVars["tools"][tool.name]["desc"] =  CUSTOM_TOOL_DESCRIPTIONS[tool.name]
        else:
            AgentOobaVars["tools"][tool.name]["desc"] = tool.description
        AgentOobaVars["tools"][tool.name]["tool"] = tool
    
def update_tool_state(tool_name, statetype, value):
    AgentOobaVars["tools"][tool_name][statetype] = value

def update_tool_description(tool_name, value):
    AgentOobaVars["tools"][tool_name]['desc'] = value

def mainloop(ostr, state):
    AgentOobaVars["processed-task-storage"] = ChromaTaskStorage()
    AgentOobaVars["processed-task-storage"].add_tasks([ostr],["MAIN OBJECTIVE"])
    yield f"<br>Thinking...<br>"
    AgentOobaVars["main-objective"] = Objective(ostr, -1, 1, state)
    while (not AgentOobaVars["main-objective"].done):
        yield f'<div class="oobaAgentOutput"><br>{AgentOobaVars["main-objective"].to_string(True)}<br>Thinking...</div>'
        AgentOobaVars["main-objective"].process_current_task()
        if AgentOobaVars["waiting-input"]:
            yield f'<div class="oobaAgentOutput"><br>{AgentOobaVars["main-objective"].to_string(True)}<br>Waiting for user input</div>'
        # Give GPU a second to breathe :)
        time.sleep(2)
        while AgentOobaVars["waiting-input"]:
            time.sleep(0.1)
    yield f'<div class="oobaAgentOutput"><br>{AgentOobaVars["main-objective"].to_string(False)}<br>Done!</div>'

def gather_agentooba_parameters(
        recursion_level,
        distance_cutoff,
        max_tasks,
        expanded_context,
        h_prefix,
        a_prefix,
        primary_directive,
        assess_ability,
        do_objective,
        split_objective,
        assess_tool,
        use_tool,
        gen_thoughts,
        summarize
        ):
        AgentOobaVars["recursion-max"] = recursion_level
        AgentOobaVars["chroma-cutoff"] = distance_cutoff
        AgentOobaVars["max-tasks"] = max_tasks
        AgentOobaVars["expanded-context"] = expanded_context
        AgentOobaVars["directives"]["Primary directive"] = primary_directive
        AgentOobaVars["directives"]["Assess ability directive"] = assess_ability
        AgentOobaVars["directives"]["Do objective directive"] = do_objective
        AgentOobaVars["directives"]["Split objective directive"] = split_objective
        AgentOobaVars["directives"]["Assess tool directive"] = assess_tool
        AgentOobaVars["directives"]["Use tool directive"] = use_tool
        AgentOobaVars["directives"]["Generate thoughts directive"] = gen_thoughts
        AgentOobaVars["directives"]["Summarize directive"] = summarize
        AgentOobaVars["human-prefix"] = h_prefix
        AgentOobaVars["assistant-prefix"] = a_prefix

def ui():
    state = gr.State({})
    with gr.Column(elem_classes="oobaAgentBase"):
        with gr.Accordion(label="Output"):
            output = gr.HTML(label="Output", value="")
        user_input = gr.Textbox(label="Goal for AgentOoba")
        with gr.Row():
            submit_button = gr.Button("Execute", variant="primary")
            cancel_button = gr.Button("Cancel")
        with gr.Accordion(label="Options", open=False):
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
                    label = "Task Similarity Cutoff (Higher = less repeat tasks, but might accidentally drop tasks)",
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
                expanded_context_toggle = gr.Checkbox(label="Expanded Context (runs out of memory at high recursion)", value = EXPANDED_CONTEXT_DEFAULT)
        with gr.Accordion(label="Tools", open = False):
            setup_tools()
            for tool_name in AgentOobaVars["tools"]:
                with gr.Row():
                    cb_active = gr.Checkbox(label=tool_name, value=False, interactive=True)
                    cb_active.change(lambda x, tn=tool_name, statetype="active" : update_tool_state(tn, statetype, x), [cb_active])
                    cb_execute = gr.Checkbox(label="Execute", value=False, interactive=True)
                    cb_execute.change(lambda x, tn=tool_name, statetype="execute": update_tool_state(tn, statetype, x), [cb_execute])
                    textbox = gr.Textbox(
                        label="Tool description (as passed to the model)",
                        interactive=True,
                        value=AgentOobaVars["tools"][tool_name]["desc"]
                    )
                    textbox.change(lambda x, tn=tool_name: update_tool_description(tn, x), [textbox])
        with gr.Accordion(label="Prompting", open = False):
            with gr.Row():
                human_prefix_input = gr.Textbox(label="Human prefix", value = HUMAN_PREFIX)
                assistant_prefix_input = gr.Textbox(label="Assistant prefix", value = ASSISTANT_PREFIX)
                human_prefix_def = gr.Textbox(visible=False, value = HUMAN_PREFIX)
                assistant_prefix_def = gr.Textbox(visible=False, value = ASSISTANT_PREFIX)
            directive_inputs = []
            directive_defaults = []
            for directive_name, directive in AgentOobaVars["directives"].items():
                directive_inputs.append(gr.TextArea(label=directive_name, value = directive))
                directive_defaults.append(gr.Textbox(visible=False, value = directive))
            prompt_inputs = directive_inputs + [human_prefix_input, assistant_prefix_input]
            prompt_defaults = directive_defaults + [human_prefix_def, assistant_prefix_def]
            reset_prompts_button = gr.Button("Reset prompts to default")
            with gr.Row():
                export_prompts_button = gr.Button("Export prompts to JSON")
                import_prompts_button = gr.Button("Import prompts from JSON")
            with gr.Row():
                exported_prompts = gr.File(interactive = False)
                imported_prompts = gr.File(interactive = True, type="binary")

    submit_event_1 = submit_button.click(
        gather_interface_values,
        inputs=gradio(list_interface_input_elements()),
        outputs=state
    ).then(
        gather_agentooba_parameters, 
        inputs=[
            recursion_level_slider, 
            distance_cutoff_slider,
            max_tasks_slider,
            expanded_context_toggle,
            human_prefix_input,
            assistant_prefix_input
            ]+directive_inputs, outputs=None
    ).then(
        mainloop, inputs=[user_input, state], outputs=output
    )

    submit_event_2 = user_input.submit(
        gather_interface_values,
        inputs=gradio(list_interface_input_elements()),
        outputs=state
    ).then(
        gather_agentooba_parameters, 
        inputs=[
            recursion_level_slider, 
            distance_cutoff_slider,
            max_tasks_slider,
            expanded_context_toggle,
            human_prefix_input,
            assistant_prefix_input
            ]+directive_inputs, outputs=None
    ).then(
        mainloop, inputs=[user_input, state], outputs=output
    )
    
    def cancel_agent():
        AgentOobaVars["main-objective"].done = True
        output.value = ""
    
    cancel_event = cancel_button.click(
        cancel_agent,
        None,
        None,
        cancels = [submit_event_1, submit_event_2]
    )
    
    reset_event = reset_prompts_button.click(
        lambda a,b,c,d,e,f,g,h,i,j: [a,b,c,d,e,f,g,h,i,j],
        inputs =  prompt_defaults,
        outputs = prompt_inputs
    )

    def make_prompt_template():
        d = AgentOobaVars["directives"].copy()
        d["human-prefix"] = AgentOobaVars["human-prefix"]
        d["assistant-prefix"] = AgentOobaVars["assistant-prefix"]
        with open("extensions/AgentOoba/prompt_template.json", "w") as f:
            f.write(json.dumps(d))
            f.flush()
        return "extensions/AgentOoba/prompt_template.json"

    def import_prompt_template(template):
        if not template:
            return [p.value for p in prompt_inputs] + [human_prefix_input.value, assistant_prefix_input.value]
        d = json.loads(template)
        return [
            d["Primary directive"],
            d["Assess ability directive"],
            d["Do objective directive"],
            d["Split objective directive"],
            d["Assess tool directive"],
            d["Use tool directive"],
            d["Generate thoughts directive"],
            d["Summarize directive"],
            d["human-prefix"],
            d["assistant-prefix"]
        ]
            
    export_event = export_prompts_button.click(
        make_prompt_template,
        inputs = None,
        outputs = [exported_prompts]
    )
    
    import_event = import_prompts_button.click(
        import_prompt_template,
        inputs = imported_prompts,
        outputs = prompt_inputs
    )

def custom_css():
    css = """
.oobaAgentBase {
  margin: auto;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
  max-width: 50%;
  font-size: 1rem;
}

.oobaAgentOutput {
  font-size: 1rem;
  list-style-type: circle;
}

.oobaAgentOutputThinking {
  font-size: 1rem;
  border-left: 2px solid orange;
  list-style-type: circle;
}

.oobaAgentOutputResource {
    font-size: 1rem;
    list-style-type: square;
}
"""
    return css.strip()