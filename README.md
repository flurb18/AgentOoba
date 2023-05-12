# AgentOoba v0.1
An autonomous AI agent extension for Oobabooga's web ui

[Screenshot](https://imgur.com/a/uapv6jd), [Sample Output](https://pastebin.com/Mp5JHEUq)

# Prerequisites
Install https://github.com/oobabooga/text-generation-webui

# Installation
1. Clone this repo inside text-generation-webui/extensions (cd /path/to/text-generation-webui/extensions && git clone https://github.com/flurb18/AgentOoba.git)
2. Inside the AgentOoba directory, copy the file `.env.example` to a file named `.env` and edit the default values if you wish. It should run fine with the default values.
3. Activate the virtual environment you used in installing the web UI.
4. Run `pip install -r requirements.txt` in the AgentOoba directory.

# Launching
1. Launch Oobabooga with the option `--extensions AgentOoba`. You can do this by editing your launch script; the line that says `python server.py (additional arguments)` should be changed to `python server.py --extensions AgentOoba (additional arguments)`. You can also just launch it normally and go to the extensions tab to enable AgentOoba, though you'll have to do this at each launch.
2. Load a model - I used TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g for all tests / designing purposes, so the other models / models of other types are untested and might not give as good results.
3. Go to the main text generation page and scroll down to see the UI.

# Info

AgentOoba is a very new project created to implement an autonomous agent in Oobabooga's web UI. It does so by making detailed requests of the underlying large language model. This agent takes a "divide-and-conquer" approach to completing tasks: if it cannot find a suitable method to complete an objective off the bat, it will try to break the task into subtasks and evaluate each subtask recursively in a breadth-first manner.

AgentOoba has a customizable prompting system: you can change how the model is prompted by editing the text of the prompts yourself in the UI. Each prompt comes with substitution variables. These are substrings such as "_TASK_" which get swapped out for other values (in the case of _TASK_, the objective at hand) before the prompt is passed to the model. Unless you plan to change the logic in the code for how the output of the model is parsed, it is inadvisable to change the "Format:" section of each prompt. This section specifies the format that we need the model to use to be able to parse its response.

The default prompts are routinely updated as I explore effective prompting methods for LLMs. If you think you have a set of prompts that work really well with a particular model or in general, feel free to share them on the Reddit threads! I am always looking for better prompts. You can export or import your set of prompts to or from a JSON file, meaning it is easy to save and share prompt templates.

# Tools

AgentOoba will support [Langchain](https://python.langchain.com/en/latest/index.html) tools. All model prompting is set up, but the input that the model generates is not yet passed to the tool. I still need to fine-tune the prompts, set up the model output parsing, update the UI, and implement context chaining (which is how the model will be able to utilize the tools output).

There are a couple of tools already included for testing purposes. You can also customize each tool's description as it is passed to the model. The tools are disabled in the UI by default; you can enable individual tools by clicking the check mark next to the tool name. The Agent will then evaluate if it can use the tool for each task.

# Credits

Entire open-source LLM community - what this movement is doing inspired me
Originally source of inspiration: https://github.com/kroll-software/babyagi4all
Oobabooga's web UI made this possible
