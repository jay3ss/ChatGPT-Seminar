try:
    import gnureadline as readline  # type: ignore
except ImportError:
    import readline

import logging
import os

import openai
from dotenv import load_dotenv

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

openai.api_key = os.environ.get("OPENAI_API_KEY")


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

system_prompt = """
You are a helpful AI assistant. Answer all user questions clearly and
concisely based on the context provided.
""".strip()

messages = [
    {"role": "system", "content": system_prompt},
]


def get_response(prompt: str, max_tokens: int = 2_000) -> None:
    messages.append({"role": "system", "content": prompt})

    stream = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    full_response = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response.append(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content, end="")
    messages.append({"role": "assistant", "content": "".join(full_response)})
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
