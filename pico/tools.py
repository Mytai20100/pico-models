import json
import re
from typing import Any, Callable, Dict, List, Optional


TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"


def build_tool_prompt(tools: List[Dict]) -> str:
    lines = ["You have access to the following tools. To call a tool, output:"]
    lines.append(f"{TOOL_OPEN}{{\"name\": \"tool_name\", \"args\": {{...}}}}{TOOL_CLOSE}")
    lines.append("\nAvailable tools:")
    for t in tools:
        lines.append(f"- {t['name']}: {t['description']}")
        if "parameters" in t:
            lines.append(f"  parameters: {json.dumps(t['parameters'])}")
    return "\n".join(lines)


def parse_tool_call(text: str) -> Optional[Dict]:
    pattern = re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE)
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def format_tool_result(result: Any) -> str:
    return f"{TOOL_RESULT_OPEN}{json.dumps(result)}{TOOL_RESULT_CLOSE}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
        self._fns: Dict[str, Callable] = {}

    def register(self, name: str, description: str, fn: Callable, parameters: Optional[Dict] = None):
        self._tools[name] = {"name": name, "description": description, "parameters": parameters or {}}
        self._fns[name] = fn

    def schema(self) -> List[Dict]:
        return list(self._tools.values())

    def call(self, name: str, args: Dict) -> Any:
        if name not in self._fns:
            return {"error": f"Unknown tool: {name}"}
        try:
            return self._fns[name](**args)
        except Exception as e:
            return {"error": str(e)}

    def run_agent_loop(self, model_fn: Callable, tokenizer, prompt: str, max_turns: int = 5, device="cpu") -> str:
        system = build_tool_prompt(self.schema())
        full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"
        history = full_prompt

        for _ in range(max_turns):
            response = model_fn(history, device=device)
            history += response

            call = parse_tool_call(response)
            if call is None:
                break

            result = self.call(call.get("name", ""), call.get("args", {}))
            result_str = format_tool_result(result)
            history += f"\n{result_str}\n"

        final = history[len(full_prompt):]
        final = re.sub(re.escape(TOOL_OPEN) + r".*?" + re.escape(TOOL_CLOSE), "", final, flags=re.DOTALL)
        final = re.sub(re.escape(TOOL_RESULT_OPEN) + r".*?" + re.escape(TOOL_RESULT_CLOSE), "", final, flags=re.DOTALL)
        return final.strip()


def default_tools() -> ToolRegistry:
    import datetime, math as _math

    reg = ToolRegistry()

    reg.register(
        "get_time",
        "Returns the current date and time.",
        lambda: {"time": datetime.datetime.now().isoformat()},
    )

    reg.register(
        "calculator",
        "Evaluates a safe math expression.",
        lambda expression: {"result": eval(expression, {"__builtins__": {}}, {k: getattr(_math, k) for k in dir(_math)})},
        {"expression": "string math expression, e.g. '2 + 2'"},
    )

    reg.register(
        "echo",
        "Echoes back the given text.",
        lambda text: {"echo": text},
        {"text": "string to echo"},
    )

    return reg
