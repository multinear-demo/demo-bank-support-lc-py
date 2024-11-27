"""
LLM observability configuration for the Bank Customer Support app.
Provides integration with various tracing tools based on environment variables:
- Logfire (https://logfire.pydantic.dev)
- Arize Phoenix (https://phoenix.arize.com)
- Simple stdout logging

This module helps in debugging and monitoring the RAG system's behavior and performance.
"""

# import os


def init_tracing():
    """
    Initialize tracing and observability tools based on environment variables.

    Configures integration with the following tools if the corresponding environment
    variables are set.
    """
    # if os.getenv("TRACE_LOGFIRE", False):
    #     print("Initializing Logfire tracing")
    #     import logfire
    #     logfire.configure()
    #     logfire.instrument_openai(Settings._llm._get_client())

    # if os.getenv("TRACE_PHOENIX", False):
    #     print("Initializing Phoenix tracing")
    #     import phoenix as px
    #     px.launch_app()
    #     set_global_handler("arize_phoenix")

    # if os.getenv("TRACE_SIMPLE", False):
    #     print("Initializing stdout tracing")
    #     set_global_handler("simple")
    pass
