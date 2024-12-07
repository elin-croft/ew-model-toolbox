import re
from typing import List

def split_with_parentheses(s) -> List[str]:
    # 使用正则表达式提取逗号分割的内容，同时忽略括号内的逗号
    pattern = r'(?:[^,(]|\([^)]*\))+'
    return re.findall(pattern, s)

def isTupleInstance(string:str):
    if string.startswith('(') and string.endswith(')'):
        return True

def kernel_size_parser(string:str):
    argList = split_with_parentheses(string)

    res = []
    if len(argList) == 1:
        size = int(argList[0])
        res.append((size, size))
        return res
    elif len(argList) > 1:
        for arg in argList:
            if isTupleInstance(arg):
                size = arg.replace("(","").replace(")","").split(",")
                assert len(size) > 1
                size = list(map(int, size))
                res.append((size[0], size[1]))
            elif arg.isdigit():
                size = int(arg)
                res.append((size, size))
        return res
