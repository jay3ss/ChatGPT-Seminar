import logging
import os
import readline

LOG_FILENAME = "/tmp/completer.log"
HISTORY_FILENAME = "/tmp/completer.hist"

logging.basicConfig(
    format="%(message)s",
    filename=LOG_FILENAME,
    level=logging.DEBUG,
)


class HistoryManager:
    def __init__(self, history_file: str) -> None:
        self.history_file = history_file
        self._load_history()

    def _load_history(self) -> None:
        if os.path.exists(self.history_file):
            readline.read_history_file(self.history_file)

    def save_history(self) -> None:
        readline.write_history_file(self.history_file)

    def get_history_items(self) -> list:
        num_items = readline.get_current_history_length() + 1
        return [readline.get_history_item(i) for i in range(1, num_items)]


class HistoryCompleter:
    def __init__(
        self, history_file: str = None, options: list[str] | None = None
    ) -> None:
        if history_file is None:
            history_file = HISTORY_FILENAME
        self.history_manager = HistoryManager(history_file)
        self.matches = []

        self._set_options(options)
        readline.set_completer(self.complete)

    def complete(self, text: str, state: int) -> str:
        response = None
        if state == 0:
            history_values = self.history_manager.get_history_items()
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

    def _set_options(self, options: list[str] | None = None) -> None:
        if options is None:
            options = ["set editing-mode vi", "tab: complete"]
        for option in options:
            readline.parse_and_bind(option)
