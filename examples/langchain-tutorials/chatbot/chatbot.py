try:
    import gnureadline as readline
except ImportError:
    import readline

import logging
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

LOG_FILENAME = "/tmp/completer.log"
HISTORY_FILENAME = "/tmp/completer.hist"

logging.basicConfig(
    format="%(message)s",
    filename=LOG_FILENAME,
    level=logging.DEBUG,
)

readline.parse_and_bind("tab: complete")
readline.parse_and_bind("set editing-mode vi")

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}


def get_history_items() -> list:
    num_items = readline.get_current_history_length() + 1
    return [readline.get_history_item(i) for i in range(1, num_items)]


class HistoryCompleter:

    def __init__(self) -> None:
        self.matches = []

    def complete(self, text: str, state: int) -> str:
        response = None
        if state == 0:
            history_values = get_history_items()
            logging.debug("history: %s", history_values)
            if text:
                self.matches = sorted(
                    h for h in history_values if h and h.startswith(text)
                )
            else:
                self.matches = []
            logging.debug("matches: %s", self.matches)
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        logging.debug("complete(%s, %s) => %s", repr(text), state, repr(response))

        return response


readline.set_completer(HistoryCompleter().complete)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


config = {"configurable": {"session_id": "abc2"}}

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(
        content="You are a good assistant, ready and willing to help to the best of your abilities."
    )
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

with_message_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
)


def get_response(message: str, language: str = "English") -> None:
    print("AI: ", end="")
    for r in with_message_history.stream(
        {
            "messages": messages + [HumanMessage(content=message)],
            "language": language,
        },
        config=config,
    ):
        print(r.content, end="")

    print()


if __name__ == "__main__":
    print("AI: Hello, how can I assist you?")
    prompt = ">>> "
    hist_file = HISTORY_FILENAME
    if os.path.exists(hist_file):
        readline.read_history_file(hist_file)
    try:
        while True:
            user_input = input(prompt)
            if user_input in ["exit", "exit()", "quit", "quit()", "bye", "bye()"]:
                break
            get_response(user_input)

    finally:
        readline.write_history_file(hist_file)
