# AgentOoba v0.3
An autonomous AI agent extension for Oobabooga's web ui

[Sample Output](https://pastebin.com/0shy8L3d)

Note: This project is still in its infancy. Right now the agent is capable of using tools and using the model's built-in capabilities to complete tasks, but it isn't great at it. It needs more context, a vague problem that I am continuously working on.

The latest update includes a change to how the flow of tasks is handled. Before, the agent would attempt to complete the task using tools as soon as it encountered it; now, it waits for child tasks to finish. What this means is you likely have to wait until the plan is fully expanded before it will start attempting objectives.

# Prerequisites
Install https://github.com/oobabooga/text-generation-webui

# Installation
1. Clone this repo inside text-generation-webui/extensions (cd /path/to/text-generation-webui/extensions && git clone https://github.com/flurb18/AgentOoba.git)
2. Inside the AgentOoba directory, copy the file `.env.example` to a file named `.env` and edit the default values if you wish. It should run fine with the default values.
3. Activate the virtual environment you used in installing the web UI.
4. Run `pip install -r requirements.txt` in the AgentOoba directory.

# Launching
1. Launch Oobabooga with the option `--extensions AgentOoba`. You can do this by editing your launch script; the line that says `python server.py (additional arguments)` should be changed to `python server.py --extensions AgentOoba (additional arguments)`. You can also just launch it normally and go to the extensions tab to enable AgentOoba, though you'll have to do this at each launch.
2. Load a model - The agent is designed to be flexible for model type, but you will have to set the human and assistant prefixes according to your model type in the Prompting section of the UI. Right now these are set for the Wizard series of model.
3. Go to the main text generation page and scroll down to see the UI.

# Info

AgentOoba is a very new project created to implement an autonomous agent in Oobabooga's web UI. It does so by making detailed requests of the underlying large language model. This agent takes a "divide-and-conquer" approach to completing tasks: if it cannot find a suitable method to complete an objective off the bat, it will try to break the task into subtasks and evaluate each subtask recursively in a breadth-first manner.

AgentOoba is designed with small-context models in mind. It's prompting system is designed to try and break up general prompts into smaller subprompts, only giving the model the context it absolutely needs for each prompt. This allows for smaller context sizes at the cost of longer execution time.

AgentOoba has a customizable prompting system: you can change how the model is prompted by editing the text of the prompts yourself in the UI. Each prompt comes with substitution variables. These are substrings such as "\_TASK\_" which get swapped out for other values (in the case of \_TASK\_, the objective at hand) before the prompt is passed to the model.

Unless you plan to change the logic in the code for how the output of the model is parsed, it is inadvisable to change the "Format:" section of each prompt. This section specifies the format that we need the model to use to be able to parse its response.

The default prompts will be routinely updated as I explore effective prompting methods for LLMs. If you have a set of prompts that work really well with a particular model or in general, feel free to share them on the Reddit threads! I am always looking for better prompts. You can export or import your set of prompts to or from a JSON file, meaning it is easy to save and share prompt templates.

# Tools

AgentOoba supports [Langchain](https://python.langchain.com/en/latest/index.html) tools. It will try to use the tool's output in future tasks as well. This is still a work in progress.

There are a couple of tools already included for testing purposes. You can also customize each tool's description as it is passed to the model. The tools are disabled in the UI by default; you can enable evaluation and execution of the tools individually by clicking the check marks next to the tool name. The Agent will then evaluate if it can use the tool for each task and will execute the tool only if allowed to.

# Credits

Entire open-source LLM community - what this movement is doing inspired me

Originally source of inspiration: https://github.com/kroll-software/babyagi4all

Oobabooga's web UI made this possible

