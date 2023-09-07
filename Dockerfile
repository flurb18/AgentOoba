FROM atinoda/text-generation-webui:default
RUN pip install langchain && \
    pip install wikipedia && \
    pip install wolframalpha
RUN pip install chromadb
COPY ./chroma_setup.py /chroma_setup.py
RUN python3 /chroma_setup.py
RUN mkdir /app/extensions/AgentOoba
COPY ./script.py /app/extensions/AgentOoba/script.py
COPY ./objective.py /app/extensions/AgentOoba/objective.py
COPY ./.env.example /app/extensions/AgentOoba/.env
