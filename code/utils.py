import pathlib
from typing import Callable


def srt_to_text(file: pathlib.Path) -> str:
    text = []
    with file.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            stripped_line = line.strip()
            should_append = stripped_line and not (
                "-->" in stripped_line or stripped_line.isdigit()
            )
            if should_append:
                text.append(stripped_line)

    return "\n".join(text)


def get_all_files_of_ext(path: str, reverse: bool = True, ext: str = "srt"):
    for file_path in sorted(pathlib.Path(path).rglob(f"*.{ext}"), reverse=reverse):
        yield file_path


def extract_text_from_file(
    path: str,
    func: Callable = srt_to_text,
    reverse: bool = True,
    args: tuple = (),
    kwargs: dict = {},
) -> str:
    return "\n".join(
        [
            func(srt_file, *args, **kwargs)
            for srt_file in get_all_files_of_ext(path, reverse=reverse)
        ]
    )


def extract_text_from_files(
    paths: str,
    func: Callable = srt_to_text,
    reverse: bool = True,
    args: tuple = (),
    kwargs: dict = {},
) -> list[str]:
    return [extract_text_from_file(path, func, reverse, args, kwargs) for path in paths]
