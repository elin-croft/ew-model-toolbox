import os

def parse_path(path: str) -> str:
    if path.endswith("/"):
        path = path[:-1]
    module_str = path.replace("/", ".").replace(".py", "")
    return module_str