from extensions.AgentOoba.script import AgentOobaVars, ooba_call
from modules.text_generation import get_encoded_length
import re
from html import escape
import uuid
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = AgentOobaVars["max-context"]/4,
    chunk_overlap  = AgentOobaVars["max-context"]/20,
    length_function = get_encoded_length
)

class Objective:
    def __init__(self, objective, task_idx, recursion_level, state, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        self.state = state
        self.tasks = []
        self.done = (recursion_level == AgentOobaVars["recursion-max"])
        self.parent_task_idx = task_idx
        self.current_task_idx = 0
        self.output = []
        self.context = {}
        self.generate_context()
        if not self.done:
            output_tasks = self.split_objective()
            self.tasks = [task for task in output_tasks if not AgentOobaVars["processed-task-storage"].task_exists(task)]
            if AgentOobaVars["verbose"] and len(self.tasks) < len(output_tasks):
                print("Tasks pruned\n", file=sys.stderr)
                before = "\n".join(output_tasks)
                after = "\n".join(self.tasks)
                print(f"Before:\n{before}\n", file=sys.stderr)
                print(f"After:\n{after}\n", file=sys.stderr)
            if len(self.tasks) == 0:
                self.done = True
            else:
                AgentOobaVars["processed-task-storage"].add_tasks(self.tasks, [uuid.uuid4().hex for task in self.tasks])

    def make_prompt(self,
                    directive,
                    include_objectives=True,
                    context_objectives=False,
                    context_resources=False,
                    context_abilities=False
                    ):
        constr=""
        context_resources = context_resources and "resources-needed" in self.context
        context_abilities = context_abilities and "abilities-needed" in self.context
        context_objectives = context_objectives and self.parent and (self.parent_task_idx > 0)
        if any([context_resources, context_abilities, context_objectives]):
            constr = "Context:\n"
            if context_resources:
                constr += f"Resources needed for completing _TASK_:\n{self.context['resources-needed']}\n"
                constr += f"Resources available:\n{self.context['resources-available'] if 'resources-available' in self.context else 'None'}\n"
            if context_abilities:
                constr += f"Abilities needed for completing _TASK_:\n{self.context['abilities-needed']}\n"
                constr += f"Abilities available:\n{self.context['abilities-available'] if 'abilities-available' in self.context else 'None'}\n"
            if context_objectives:
                constr += f"The following is a list of objectives that have already been completed:\n"
                constr += "\n".join([f"Objective {self.recursion_level-1}, Task {i+1}: {self.parent.tasks[i].objective}" for i in range(self.parent_task_idx)])
                constr += "\n"
            constr += "\n"
        directive = "\n".join([line.strip() for line in (directive.split("\n") if "\n" in directive else [directive])])
        directive = directive.replace("_TASK_", f"Objective {self.recursion_level}").strip()
        constr = constr.replace("_TASK_", f"Objective {self.recursion_level}")
        objstr = f"Remember these objectives:\n{self.prompt_objective_context()}\n\n" if include_objectives else ""
        return f"{AgentOobaVars['human-prefix']}\n{AgentOobaVars['directives']['Primary directive']}\n\n{objstr}{constr}Instructions:\n{directive}\n\n{AgentOobaVars['assistant-prefix']}"

    def assess_model_ability(self):
        directive = AgentOobaVars["directives"]["Assess ability directive"]
        prompt = self.make_prompt(directive, include_objectives=True, context_abilities=True, context_resources=True)
        response = ooba_call(prompt, self.state).strip()
        return 'yes' in response.lower()

    def do_objective(self):
        directive = AgentOobaVars["directives"]["Do objective directive"]
        response = ooba_call(self.make_prompt(directive, include_objectives=True, context_abilities=True, context_resources=True), self.state).strip()
        return response

    def generate_context(self):
        self.context["resources-available"]="None"
        init_abilities="""
- Following Instructions: You follow instructions exceptionally well and pay close attention to them.
- Generating Text: You are an AI and can generate text. You can use this ability for tasks such as writing, summarizing, making decisions, answering questions, and developing plans.
- Using Tools: You can use any tools that are available to you.
        """
        self.context["abilities-available"]=init_abilities.strip()
        directive = AgentOobaVars["directives"]["Generate thoughts directive"]
        response = ooba_call(self.make_prompt(directive, include_objectives=True), self.state).strip()
        context_regex = re.compile('Resources: (.+)\nAbilities: (.+)',re.DOTALL)
        match = context_regex.search(response)
        if not match:
            return
        g = match.groups()
        self.context["resources-needed"]=g[0]
        self.context["abilities-needed"]=g[1]
    
    def split_objective(self):
        directive = AgentOobaVars["directives"]["Split objective directive"].replace("_MAX_TASKS_", str(AgentOobaVars["max-tasks"]))
        prompt = self.make_prompt(directive, include_objectives=True, context_objectives=True)
        response = ooba_call(prompt, self.state).strip()
        task_list_regex = re.compile('((^|\n)[\d]+\.)(.*?)(?=(\n[\d]+\..*)|($))', re.DOTALL)
        match = task_list_regex.search(response)
        task_list = []
        while match:
            g = match.groups()
            task_list.append(g[2].strip())
            if g[3]:
                match = task_list_regex.search(g[3])
            else:
                break
        return task_list
            
    def assess_tools(self):
        for tool_name in AgentOobaVars["tools"]:
            if AgentOobaVars["tools"][tool_name]["active"]:
                tool_str = f"Tool name: {tool_name}\nTool description: {AgentOobaVars['tools'][tool_name]['desc']}"
                directive = AgentOobaVars["directives"]["Assess tool directive"].replace("_TOOL_NAME_", tool_name)
                old = self.context["resources-available"]
                self.add_resource_no_summary(f"You have the following tool available to you:\n{tool_str}")
                prompt = self.make_prompt(directive, include_objectives=True, context_resources=True)
                if 'yes' in ooba_call(prompt, self.state).strip().lower():
                    directive = AgentOobaVars["directives"]["Use tool directive"].replace("_TOOL_NAME_", tool_name)
                    prompt = self.make_prompt(directive, include_objectives=True, context_resources=True)
                    response = ooba_call(prompt, self.state).strip()
                    negative_responses = ["i cannot", "am unable"]
                    if not any([neg in response.lower() for neg in negative_responses]):
                        self.context["resources-available"]=old
                        return True, AgentOobaVars["tools"][tool_name]["tool"], response
                self.context["resources-available"]=old
        return False, None, None
    
    def prompt_objective_context(self):
        reverse_context = []
        p_it = self
        r = self.recursion_level
        while p_it.parent:
            child = p_it
            p_it = p_it.parent
            if AgentOobaVars["expanded-context"]:
                parent_task_list_str = "\n".join([f"Objective {r-1}, Task {str(i+1)}: {p_it.tasks[i] if isinstance(p_it.tasks[i], str) else p_it.tasks[i].objective}" for i in range(len(p_it.tasks))])
                reverse_context.append(f"We have developed the following numbered list of tasks that one must complete to achieve Objective {r-1}:\n{parent_task_list_str}\n\nThe current task that we are at among these is Objective {r-1}, Task {p_it.current_task_idx+1}. We will refer to Objective {r-1}, Task {p_it.current_task_idx+1} as Objective {r}.")
            else:
                reverse_context.append(f"In order to complete Objective {r-1}, one must complete Objective {r}. Objective {r} is: {child.objective}")
            r -= 1
        assert r == 1
        reverse_context.append(f"Objective 1 is: {p_it.objective}")
        reverse_context.reverse()
        return "\n".join(reverse_context)

    def add_resource(self, resource):
        i = 0
        while get_encoded_length(resource) > (AgentOobaVars["max-context"] / 4) and i < AgentOobaVars["max-summaries"]:
            i += 1
            docs = text_splitter.create_documents([resource])
            summaries = []
            for doc in docs:
                directive = AgentOobaVars["directives"]["Summarize directive"].replace("_TEXT_", doc)
                prompt = self.make_prompt(directive, include_objectives=False)
                summaries.append(ooba_call(prompt, self.state).strip())
            resource = "\n\n".join(summaries)
        final_length = get_encoded_length(resource)
        if final_length < AgentOobaVars["max-context"]:
            if final_length > (AgentOobaVars["max-context"]/4):
                directive = AgentOobaVars["directives"]["Summarize directive"].replace("_TEXT_", resource)
                prompt = self.make_prompt(directive, include_objectives=False)
                resource = ooba_call(prompt, self.state).strip()
            self.add_resource_no_summary(resource)

    def add_resource_no_summary(self, resource):
        if not "resources-available" in self.context or self.context["resources-available"] == "None":
            self.context["resources-available"] = resource
        else:
            self.context["resources-available"] += f"\n{resource}"

    def try_objective(self):
        tool_found, tool, tool_input = self.assess_tools()
        if tool_found:
            if (AgentOobaVars["tools"][tool.name]["execute"]):
                used_tool_str = f"TOOL USED: \"{tool.name}\"\nINPUT: \"{tool_input}\"\nOUTPUT: \"{tool.run(tool_input)}\""
                self.output.append(used_tool_str)
                if self.parent:
                    self.parent.add_resource(used_tool_str)
            else:
                self.output.append(f"TOOL FOUND: \"{tool.name}\"\nINPUT: \"{tool_input}\"")
        if self.assess_model_ability():
            response = self.do_objective()
            negative_responses = ["i cannot", "i am unable", "i'm unable"]
            if not any([neg in response.lower() for neg in negative_responses]):
                self.output.append(f"MODEL OUTPUT {response}")
                if self.parent:
                    self.parent.add_resource(response)

    def process_current_task(self):
        if self.current_task_idx == len(self.tasks):
            self.current_task_idx = 0
            if all([(isinstance(task, str) or task.done) for task in self.tasks]):
                self.try_objective()
                self.done = True
        if not self.done:
            current_task = self.tasks[self.current_task_idx]
            if isinstance(current_task, str):
                self.tasks[self.current_task_idx] = Objective(
                    current_task,
                    self.current_task_idx,
                    self.recursion_level + 1,
                    self.state,
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
        html_string = f'OBJECTIVE: {escape(self.objective)}<ul class="oobaAgentOutput">'
        for task in self.tasks:
            thinking = False
            p_it = self
            task_idx = p_it.tasks.index(task) if isinstance(task, str) else task.parent_task_idx
            while ((p_it.current_task_idx % len(p_it.tasks)) == task_idx):
                if not p_it.parent:
                    thinking = True
                    break
                task_idx = p_it.parent_task_idx
                p_it = p_it.parent
            task_disp_class = "oobaAgentOutputThinking" if thinking and select else "oobaAgentOutput"
            if isinstance(task, str):
                html_string += f'<li class="{task_disp_class}">{escape(task)}</li>'
            else:
                html_string += f'<li class="{task_disp_class}">{task.to_string(select)}</li>'
        for out in self.output:
            html_string += f'<li class="oobaAgentOutputResource">{escape(out)}</li>'
        html_string += "</ul>"
        return html_string