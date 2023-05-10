# AgentOoba v0.1
An autonomous AI agent extension for Oobabooga's web ui

[Screenshot](https://imgur.com/a/uapv6jd), [Sample Output](https://pastebin.com/Mp5JHEUq)

Prerequisites:
Install https://github.com/oobabooga/text-generation-webui

Installation:
1. Clone this repo inside text-generation-webui/extensions (cd /path/to/text-generation-webui/extensions && git clone https://github.com/flurb18/AgentOoba.git)
2. Inside the AgentOoba directory, copy the file `.env.example` to a file named `.env` and edit the default values if you wish. It should run fine with the default values.
3. Activate the virtual environment you used in installing the web ui.
4. Run `pip install -r requirements.txt` in the AgentOoba directory.

Launching:
1. Launch Oobabooga with the option `--extensions AgentOoba`. You can do this by editing your launch script; the line that says `python server.py (additional arguments)` should be changed to `python server.py --extensions AgentOoba (additional arguments)`. You can also just launch it normally and go to the extensions tab to enable AgentOoba, though you'll have to do this at each launch.
2. Load a model - I used TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g for all tests / designing purposes, so the other models / models of other types are untested and might not give as good results.
3. Go to the main text generation page and scroll down to see the UI.

Originally inspired by https://github.com/kroll-software/babyagi4all