"""Search-only experiment identity, kept separate from cited source text."""
import re


def path_context(path: str) -> str:
    return "Experiment source path: " + path + "\n" + re.sub(r"[_/.-]+", " ", path)
