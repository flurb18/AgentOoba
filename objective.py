from extensions.AgentOoba.script import AgentOobaVars, ooba_call
import re
from html import escape
import uuid
import sys

class Objective:
    def __init__(self, objective, task_idx, recursion_level, parent=None):
        self.objective = objective
        self.parent = parent
        self.recursion_level = recursion_level
        self.tasks = []
        self.done = (recursion_level == AgentOobaVars["recursion-max"])
        self.parent_task_idx = task_idx
        self.current_task_idx = 0
        self.output = ""
        self.context = {}
        self.generate_context()
        if self.assess_model_ability():
            response = self.do_objective()
            negative_responses = ["i cannot", "i am unable", "i'm unable"]
            if not any([neg in response.lower() for neg in negative_responses]):
                self.done = True
                self.output= f"MODEL OUTPUT {response}"
                return
        tool_found, tool, tool_input = self.assess_tools()
        if tool_found:
            self.done = True
            if (AgentOobaVars["tools"][tool.name]["execute"]):
                output = f"I executed the tool \"{tool.name}\" with the input:{tool_input}\nThe tool returned these results:\n"
                output += tool.run(tool_input);
                self.output = output
            else:
                self.output = f"TOOL FOUND {tool.name} input={tool_input}"
        else:
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
                    context_resources=False,
                    context_abilities=False
                    ):
        constr=""
        if any([context_resources, context_abilities]):
            constr = "Context:\n"
            if context_resources and "resources-needed" in self.context:
                constr += f"Resources needed for completing _TASK_:\n{self.context['resources-needed']}\n"
                constr += f"Resources available:\n{self.context['resources-available'] if 'resources-available' in self.context else 'None'}\n"
            if context_abilities and "abilities-needed" in self.context:
                constr += f"Abilities needed for completing _TASK_:\n{self.context['abilities-needed']}\n"
                constr += f"Abilities available:\n{self.context['abilities-available'] if 'abilities-available' in self.context else 'None'}\n"
        directive = "\n".join([line.strip() for line in (directive.split("\n") if "\n" in directive else [directive])])
        directive = directive.replace("_TASK_", f"Objective {self.recursion_level}").strip()
        constr = constr.replace("_TASK_", f"Objective {self.recursion_level}")
        objstr = f"Remember these objectives:\n{self.prompt_objective_context()}\n\n" if include_objectives else ""
        return f"{AgentOobaVars['human-prefix']}\n{AgentOobaVars['directives']['Primary directive']}\n{objstr}{constr}\nInstructions:\n{directive}\n\n{AgentOobaVars['assistant-prefix']}"

    def assess_model_ability(self):
        directive = AgentOobaVars["directives"]["Assess ability directive"]
        prompt = self.make_prompt(directive, include_objectives=True, context_abilities=True, context_resources=True)
        response = ooba_call(prompt).strip()
        return 'yes' in response.lower()

    def do_objective(self):
        directive = AgentOobaVars["directives"]["Do objective directive"]
        response = ooba_call(self.make_prompt(directive, include_objectives=True, context_resources=True)).strip()
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
        response = ooba_call(self.make_prompt(directive, include_objectives=True)).strip()
        context_regex = re.compile('Resources: (.+)\nAbilities: (.+)',re.DOTALL)
        match = context_regex.search(response)
        if not match:
            return
        g = match.groups()
        self.context["resources-needed"]=g[0]
        self.context["abilities-needed"]=g[1]
    
    def split_objective(self):
        directive = AgentOobaVars["directives"]["Split objective directive"].replace("_MAX_TASKS_", str(AgentOobaVars["max-tasks"]))
        prompt = self.make_prompt(directive, include_objectives=True)
        response = ooba_call(prompt).strip()
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
                self.context["resources-available"]=f"You have the following tool available to you:\n{tool_str}"
                prompt = self.make_prompt(directive, include_objectives=True, context_resources=True, context_abilities=True)
                self.context["resources-available"]="None"
                if 'yes' in ooba_call(prompt).strip().lower():
                    directive = AgentOobaVars["directives"]["Use tool directive"].replace("_TOOL_NAME_", tool_name)
                    prompt = self.make_prompt(directive, include_objectives=True)
                    response = ooba_call(prompt).strip()
                    negative_responses = ["i cannot", "am unable"]
                    if not any([neg in response.lower() for neg in negative_responses]):
                        return True, AgentOobaVars["tools"][tool_name]["tool"], response
        return False, None, None
    
    def prompt_objective_context(self):
        reverse_context = []
        p_it = self
        r = self.recursion_level
        #if include_parent_tasks and self.parent:
        #    task_list_str = "\n".join([(task if isinstance(task, str) else task.objective) for task in self.parent.tasks])
        #    reverse_context.append(f"The following is a list of tasks that have already been processed:\n{task_list_str}")
        #reverse_context.append(f"Objective {r} is the current task we are working on.")
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
        out = f'OBJECTIVE: {escape(self.objective)}<ul class="oobaAgentOutput">'
        if self.output:
            out += f'<li class="oobaAgentOutput">{escape(self.output)}</li>'
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
                    out += f'<li class="{task_disp_class}">{escape(task)}</li>'
                else:
                    out += f'<li class="oobaAgentOutput">{task.to_string(select)}</li>'
        out += "</ul>"
        return out