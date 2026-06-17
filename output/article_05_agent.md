# （四）赋予手脚：从Function Calling到Multi-Agent

> 让模型调API、搜网页、替你干活

---

上一篇你的RAG系统能回答关于文档的问题了。你问"实验结论是什么"，它翻文档给你答案。但你让它"把实验数据汇总成表格，存到Excel里，再发邮件给导师"——它不会。

因为RAG只能读和写文字。它没有"做"的能力。

把"做"的能力赋予模型，核心机制叫 **Function Calling**（工具调用）：模型根据你的问题，自己决定调用哪个函数、传什么参数。搜天气？它调天气API。查数据库？它执行SQL。发邮件？它调发邮件接口。

这篇不讲理论框架——我们从最底层开始，先看 LangChain 框架的 ReAct Agent，再手写一个自己的，最后用多 Agent 协作完成更复杂的任务。

所有搜索工具统一使用百度搜索（`baidusearch` 库），不需要申请任何 API Key。

---

## 工具准备

本教程所有 Agent 共用同一套工具。下面先定义搜索工具（百度搜索，免 Key）和计算器，后续的代码块直接引用。

```python
# ── 工具定义：search_web（Baidu，免 Key）+ calculator ──
# pip install requests baidusearch (如未装)

import ast
import operator
from baidusearch.baidusearch import search as baidu_search

def safe_eval(expr: str):
    """安全计算器（替代 eval）"""
    _OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv}
    node = ast.parse(expr.strip(), mode="eval")
    def _eval(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -_eval(n.operand)
        raise ValueError(f"不支持: {type(n).__name__}")
    return _eval(node.body)

def search_web(query: str) -> str:
    """搜索互联网（使用百度搜索，不需要 API Key）"""
    try:
        results = baidu_search(query, num_results=3)
        if not results:
            return f"未找到结果: {query}"
        out = []
        for r in results:
            title = r.get("title", "").strip()
            abstract = r.get("abstract", "").strip()
            abstract = " ".join(abstract.split())
            out.append(f"• {title}\n  {abstract[:200]}")
        return "\n".join(out)
    except Exception as e:
        return f"搜索出错: {e}"

def calculator(expression: str) -> str:
    try:
        return f"计算结果: {safe_eval(expression)}"
    except Exception as e:
        return f"计算出错: {e}"

# 注册到工具字典
TOOLS = {
    "search_web": {"fn": search_web, "desc": "搜索互联网获取最新信息", "params": {"query": "搜索关键词"}},
    "calculator": {"fn": calculator, "desc": "执行数学计算（加减乘除）", "params": {"expression": "数学表达式, 如 2025-1990"}},
}
```

> **生产环境中的搜索方案**：本教程使用 `baidusearch` 爬取百度搜索结果，完全零成本、不需要注册任何 API Key，方便你在本地跑通实验。但这本质是网页爬虫，存在三个问题：(1) 违反搜索服务商的使用条款；(2) 结果格式不稳定，百度改版就可能失效；(3) 延迟高、没有 SLA 保障。
>
> 生产环境的搜索工具应该接入正式的搜索 API，常见的方案有（以下三个任选一个即可）：
>
> - **Bing Search API**（推荐首选）：Azure Portal 申请免费层（F1 定价层），每月 1000 次免费调用，超出后每千次约 \\$1。国内可访问，文档完善，比 Google 更容易申请。接入只需把上面的 `search_web` 内部改成 `requests.get` 调 Bing 接口。
> - **SerpAPI**：聚合 Google / Bing / 百度等多家搜索引擎的统一 API，免费层每月 100 次。优势是一个接口对接多家搜索源，适合需要对比多源结果的场景。
> - **Tavily**：专为 AI Agent 设计的搜索 API，返回的是"适合 LLM 阅读"的精炼结果（去广告去重提炼），而非原始 HTML。和本教程"搜索→摘要→推理"的模式天然匹配，是目前 Agent 社区的主流选择。
>
> 如果项目是对外发布的产品，直接买搜索 API 是最省心的选择。如果是内部工具流量不大，免费层通常足够——每月 1000 次搜索平均下来每天 30 多次，个人使用绰绰有余。

---

## 核心机制：Function Calling 只有四步

不管用哪种框架，底层逻辑都一样：

1. 定义工具的 schema（这个函数叫什么、需要什么参数）
2. 把 schema 传给模型（模型知道它有什么工具可用）
3. 模型返回 function_name + args（模型决定要调用哪个工具）
4. 执行函数，把结果返回给模型（模型基于结果生成最终回答）

---

> **生产环境安全提示**：Function Calling 让模型能够执行工具函数。本教程中的工具（搜索、计算器）是无害的，但如果在生产环境中不加限制地暴露文件操作、数据库写入、发送邮件等工具给模型，攻击者可能通过 prompt injection 诱导模型调用危险操作。**生产环境必须给每个工具加权限控制**——谁可以调用、调用频次限制、操作范围白名单，缺一不可。

> **建议 temperature=0**：工具调用和创意写作不同——你希望模型稳定地选择正确的工具、输出正确的参数，而不是发挥创造力。设 `temperature=0` 可以显著减少"幻觉工具名"和"参数格式跑偏"的问题。后面代码示例统一使用 `temperature=0`。

> **结构化 JSON 输出 ≠ Function Calling**：这两个概念常被混淆。Function Calling 的核心是"模型决定调用一个工具并执行它"——**有副作用**，会触发真实操作（查数据库、发邮件、执行代码）。而结构化输出（`response_format: {type: "json"}`）只是"让模型输出格式化的 JSON"——**没有副作用**，模型只改变了输出格式，不会触发任何外部操作。如果你只是想让模型返回结构化的数据（比如从一段文字中提取姓名、日期、金额），用结构化输出就够了，不需要声明工具。反过来，只有当你需要模型触发实际操作（搜索网页、写数据库、发送请求）时，才需要用 Function Calling。

---

接下来看用 LangChain 框架搭建的 ReAct Agent——它替我们做了上面 4 步中的循环管理：

```python
from langchain_community.llms import LlamaCpp
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from baidusearch.baidusearch import search as baidu_search

MODEL_PATH = r"path/to/your/model.gguf"

llm = LlamaCpp(model_path=MODEL_PATH, temperature=0, max_tokens=1024, n_ctx=4096, verbose=False)

@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    try:
        results = baidu_search(query, num_results=3)
        if not results:
            return f"未找到关于 '{query}' 的结果"
        return "\n".join(f"{i}. {r['title']}\n   {r.get('snippet', r.get('abstract',''))[:200]}" for i, r in enumerate(results, 1))
    except Exception as e:
        return f"搜索出错: {e}"

tools = [search_web]

template = """请根据以下工具回答问题。

可用工具:
{tools}

必须严格按以下格式回答（每行一个字段，不要合并）：

Question: 用户的问题
Thought: 你的思考过程
Action: 工具名称（必须是 [{tool_names}] 之一）
Action Input: 传给工具的参数
Observation: 工具返回的结果（由系统自动填入，你不要写这一行）
... 可以重复 Thought/Action/Action Input/Observation
Thought: 我已经知道最终答案了
Final Answer: 最终回答

重要规则：
1. Action 和 Action Input 必须分两行写，不能写在同一行！
2. 输出 Final Answer 后必须立即停止
3. 只回答用户提出的这一个问题

开始！

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True)

result = agent_executor.invoke({"input": "2025年诺贝尔物理学奖得主是谁？"})
print(result["output"])
```

**运行结果：**

```
> Entering new AgentExecutor chain...
需要查找2025年诺贝尔物理学奖的获奖者。
Action: search_web
Action Input: 2025年诺贝尔物理学奖得主是谁？

2025年诺贝尔物理学奖揭晓
2025年10月7日，瑞典皇家科学院决定将2025年诺贝尔物理学奖授予约翰·克拉克、
米歇尔·H·德沃雷与约翰·M·马丁尼斯，以表彰他们"发现电路中的宏观量子力学隧道效应和能量量子化"。

Final Answer: 2025年诺贝尔物理学奖得主是约翰·克拉克、麦克·H·德沃雷特和约翰·M·马蒂尼。
```

LangChain 帮你管理了"思考→行动→观察"的循环。每次模型输出 `Action:`，框架自动执行对应的工具，把观察结果拼回去，直到模型输出 `Final Answer`。

---

### 手写 ReAct Agent（真实 LLM 调用）

上面用了 LangChain 框架帮你管理循环。这次手写一个——每一步的思考、工具选择、观察反馈都由你自己控制的代码驱动，让你看清 ReAct 内部到底发生了什么。核心循环只有 5 行：

```
while i < max_iterations:
    response = llm(messages)            # 1. 模型选择工具
    parsed = parse(response)             # 2. 从文字中提取 Action
    result = tools[parsed.action](args)  # 3. 执行工具
    messages += Observation              # 4. 反馈结果给模型
```

工具定义使用前面已经定义好的 `TOOLS` 字典。

> **注意：Token 消耗**：每轮 ReAct 循环（思考→行动→观察）都会把之前的完整对话历史 + 新的观察结果重新喂给模型，上下文不断膨胀。4 轮交互单次请求就能消耗 2000-4000 tokens。如果设了 `max_iterations=10`，一轮完整 Agent 调用可能吃掉上万 tokens。这是 Agent 推理的固有成本——不是 bug，但设计任务时要心中有数，避免让 Agent 做无意义的反复尝试。

```python
from langchain_community.llms import LlamaCpp
import json, re

MODEL_PATH = r"path/to/your/model.gguf"
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=4096, temperature=0, verbose=False)

TOOL_DESC = "\n".join(
    f"- {name}: {info['desc']}, 参数: {json.dumps(info['params'], ensure_ascii=False)}"
    for name, info in TOOLS.items()
)

SYSTEM_PROMPT = f"""你是一个能使用工具的AI助手。你有以下工具可用：

{TOOL_DESC}

请严格用中文按以下格式回答，每次只输出一步：

如果你需要调用工具：
思考：你的推理过程
动作：工具名称
参数：工具的参数

如果你已经知道答案：
思考：我已经知道答案了
最终答案：你的最终回答

重要规则：
- 思考、动作、参数必须各占一行
- 不要自己写观察结果，系统会自动填入
- 一次只能输出一个动作或一个最终答案
- 调用工具后等待观察结果，再决定下一步
- 搜索词必须使用中文，不要翻译成英文
- 最终答案输出后立即停止"""

def call_llm(messages):
    prompt = ""
    for m in messages:
        role = m["role"].capitalize()
        prompt += f"<|{role}|>\n{m['content']}<|end|>\n"
    prompt += "<|Assistant|>\n"
    return llm.invoke(prompt)

def parse_react(text):
    colon = r"[：:]"
    action = re.search(rf"动作{colon}\s*(\w+)", text)
    inp = re.search(rf"参数{colon}\s*(.+)", text)
    if action and inp:
        return ("action", action.group(1), inp.group(1).strip().strip("\"'"))
    m = re.search(rf"最终答案{colon}\s*(.*)", text, re.DOTALL)
    if m:
        return ("final", m.group(1).strip())
    # fallback 英文标签
    action = re.search(r"Action:\s*(\w+)", text)
    inp = re.search(r"Action Input:\s*(.+)", text)
    if action and inp:
        return ("action", action.group(1), inp.group(1).strip().strip("\"'"))
    m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if m:
        return ("final", m.group(1).strip())
    return ("error", text)

def normalize_arg(raw: str):
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return next(v for v in json.loads(raw).values() if isinstance(v, str))
        except Exception:
            pass
    return raw

def react_agent(question, max_iterations=10):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    seen_searches = set()
    for i in range(max_iterations):
        response = call_llm(messages)
        kind = parse_react(response)

        if kind[0] == "final":
            return kind[1]
        if kind[0] == "error":
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "格式错误！请按格式：思考：…\n动作：工具名称\n参数：参数值"})
            continue

        _, tool_name, tool_input = kind
        tool_input = normalize_arg(tool_input)

        if tool_name == "search_web" and tool_input in seen_searches:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"已搜过，换个搜索词"})
            continue
        seen_searches.add(tool_input)
        
        result = TOOLS[tool_name]["fn"](tool_input)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"观察结果: {result}"})

# 运行
print(react_agent("2025年获得诺贝尔文学奖的是谁？年龄多大了？"))
```

**运行结果（4轮完整交互）：**

```
问题: 2025年获得诺贝尔文学奖的是谁？年龄多大了？

─── 第 1 轮 ───
[模型回复]
思考：我需要查找2025年获得诺贝尔文学奖的获奖者及其年龄。
动作：search_web
参数：{"query": "2025年获得诺贝尔文学奖的人和年龄"}
[调用] search_web(2025年获得诺贝尔文学奖的人和年龄)
[观察] • 2025诺贝尔文学奖揭晓!71岁匈牙利作家是中国诗人李白的粉丝
  2025年10月9日，71岁的匈牙利作家拉斯洛·克拉斯纳霍尔凯获得本年度诺贝尔文学奖

─── 第 2 轮 ───
[模型回复]
思考：需要查找他的出生年份以计算年龄。
动作：search_web
参数：{"query": "拉斯洛·克拉斯纳霍尔凯出生年份"}
[调用] search_web(拉斯洛·克拉斯纳霍尔凯出生年份)
[观察] • 拉斯洛·克拉斯纳霍尔凯,1954年出生于匈牙利久洛市

─── 第 3 轮 ───
[模型回复]
思考：出生于1954年，计算2025年的年龄。
动作：calculator
参数：{"expression": "2025-1954"}
[调用] calculator(2025-1954)
[观察] 计算结果: 71

─── 第 4 轮 ───
[模型回复]
思考：年龄是71岁。
最终答案：拉斯洛·克拉斯纳霍尔凯在2025年获得诺贝尔文学奖时的年龄为71岁。

[OK] [最终答案] 拉斯洛·克拉斯纳霍尔凯在2025年获得诺贝尔文学奖时的年龄为71岁。
```

可以看到，手写 ReAct 版本的 Agent 在 4 轮交互中完成了搜索获奖者、搜索出生年份、计算年龄、输出答案的完整流程。所有格式标签都用中文，搜索词也是中文。

> **关于搜索工具**：当前 `search_web` 使用 `baidusearch` 库（爬取百度搜索结果），不需要 API Key。这仅用于本地实验，详见上方[生产环境的搜索方案](#生产环境中的搜索方案)一节。如果百度改版导致搜索失效，只改 `search_web` 函数内部实现即可，ReAct 循环不需要动。
>
> **关于计算器异常**：模型在 `参数` 中常以 JSON 格式输出（如 `{"expression": "2025-1958"}`），`normalize_arg` 函数会自动检测并解包。如果计算器报错，检查 `normalize_arg` 是否覆盖了你的模型输出格式。

---

### Multi-Agent 协作示例

多个 Agent 各司其职，适合复杂任务。以下设计：

- **Coder (Phi-4 Q3_K_M)** — 独立写代码
- **Coder (Phi-4 Q4_K_M)** — 独立写代码（并行执行）
- **Technical Reviewer (Phi-4 Q4_K_M)** — 统一审查两份代码，给出简要建议
- **Tech Writer (Phi-4 Q4_K_M)** — 比较打分，选出推荐方案

> ⚠ **量化版本差异 vs. 任务难度**：Q3 和 Q4 是同一个模型的轻度/重度量化版。简单的任务（如"循环求和"）两者输出几乎没有区别。下面选了一个**中等难度**的任务——解析混合格式文本数据——既能让差异显现，又不至于让 Q3 写崩。

```python
# ── 并行 Multi-Agent 协作：两个 Coder 同时写 → 统一审查 → 比较打分 ──

from langchain_community.llms import LlamaCpp
from concurrent.futures import ThreadPoolExecutor
import json, textwrap

MODEL_Q4 = r"path/to/phi4-q4.gguf"
MODEL_Q3 = r"path/to/phi4-q3.gguf"

def make_llm(path):
    return LlamaCpp(model_path=path, n_ctx=4096, temperature=0, verbose=False)

llm_q4 = make_llm(MODEL_Q4)
llm_q3 = make_llm(MODEL_Q3)

def call_llm(llm, system_prompt, user_message):
    prompt = f"<|System|>\n{system_prompt}<|end|>\n<|User|>\n{user_message}<|end|>\n<|Assistant|>\n"
    return llm.invoke(prompt)

# ── 任务定义 ──
TASK = """写一个 Python 函数 parse_logs(text: str) -> dict，解析以下格式的服务器日志。

日志格式示例：
[2025-06-01 08:12:33] [INFO] [auth] 用户登录成功 user=admin
[2025-06-01 08:15:47] [WARN] [db] 连接池使用率 85%
[2025-06-01 08:16:02] [ERROR] [auth] 无效登录尝试 user=unknown
[2025-06-01 08:20:11] [INFO] [api] GET /users 200 45ms
[2025-06-01 08:21:05] [ERROR] [db] 查询超时: SELECT * FROM orders
[2025-06-01 08:22:30] [WARN] [api] 响应时间超过 1000ms: /reports

要求：
1. 解析每一行，提取：时间戳、日志级别(INFO/WARN/ERROR)、模块名、消息内容
2. 返回 dict，包含：
   - total_lines: 总行数
   - by_level: 各级别计数 {"INFO": 2, "WARN": 2, "ERROR": 2}
   - by_module: 各模块计数 {"auth": 2, "db": 2, "api": 2}
   - error_lines: 所有 ERROR 级别的完整行列表
3. 只输出代码本身，不要解释。"""

print("=" * 60)
print(f"任务:\n{textwrap.indent(TASK, '  ')}")
print("=" * 60)

# ── 阶段 1：并行 Coding ──
print("\n>>> [阶段1] 两个 Coder 并行写代码...\n")

AGENT_CODER = "你是一个 Python 专家。根据用户需求写出可直接运行的代码。只输出代码本身，不要额外解释。"

with ThreadPoolExecutor(max_workers=2) as pool:
    fut_q3 = pool.submit(call_llm, llm_q3, AGENT_CODER, TASK)
    fut_q4 = pool.submit(call_llm, llm_q4, AGENT_CODER, TASK)
    code_q3 = fut_q3.result()
    code_q4 = fut_q4.result()

print(f">>> [Coder Q3_K_M]\n{code_q3}\n")
print(f">>> [Coder Q4_K_M]\n{code_q4}\n")

# ── 阶段 2：统一审查 ──
print(">>> [阶段2] Technical Reviewer 审查两份代码...\n")

REVIEWER_PROMPT = """你是一个资深的代码审查者。检查以下两段代码，从安全性、正确性、健壮性、可读性四个维度给出简要评价。

对每段代码列出 1-2 个最关键的问题和建议。保持简洁。"""

review_input = f"【代码 A — Q3_K_M 版本】\n{code_q3}\n\n【代码 B — Q4_K_M 版本】\n{code_q4}"
review = call_llm(llm_q4, REVIEWER_PROMPT, review_input)
print(f">>> [Reviewer]\n{review}\n")

# ── 阶段 3：比较打分 ──
print(">>> [阶段3] Tech Writer 比较打分...\n")

JUDGE_PROMPT = """你是一个技术写手和技术经理。比较以下两段代码和审查意见，从四个维度打分（1-10分）：

1. 正确性 - 能否正确处理示例中的各种日志行
2. 可读性 - 代码结构、命名、注释是否清晰  
3. 效率 - 实现是否简洁高效
4. 健壮性 - 边界情况和异常处理

输出格式（严格按以下 markdown 格式）：

| 维度 | 代码A (Q3) | 代码B (Q4) |
|---|---|---|
| 正确性 | X | X |
| 可读性 | X | X |
| 效率 | X | X |
| 健壮性 | X | X |

**总分**: 代码A = XX, 代码B = XX

**推荐方案**: 代码A / 代码B
**理由**: 一两句话说明推荐原因。"""

judge_input = f"任务要求：\n{TASK}\n\n【代码 A — Q3_K_M】\n{code_q3}\n\n【代码 B — Q4_K_M】\n{code_q4}\n\n【审查意见】\n{review}"
judgment = call_llm(llm_q4, JUDGE_PROMPT, judge_input)
print(f">>> [Judge]\n{judgment}\n")

print("=" * 60)
print("Multi-Agent 并行协作完成！")
print(f"流程: Q3 Coder ─┐")
print(f"              ├── Reviewer → Judge")
print(f"      Q4 Coder ─┘")
```

**实际运行输出：**

```
============================================================
任务:
  写一个 Python 函数 parse_logs(text: str) -> dict...
============================================================

>>> [阶段1] 两个 Coder 并行写代码...

写代码部分省略

>>> [阶段2] Technical Reviewer 审查两份代码...

>>> [Reviewer]

审查意见部分省略

>>> [阶段3] Tech Writer 比较打分...

>>> [Judge]
| 维度 | 代码A (Q3) | 代码B (Q4) |

...     ...         ...

**总分**: 代码A = 27, 代码B = 31

**推荐方案**: 代码B
**理由**: 代码B 使用正则限定级别和模块范围，defaultdict 简化计数逻辑，整体更健壮。

============================================================
Multi-Agent 并行协作完成！
流程: Q3 Coder ─┐
              ├── Reviewer → Judge
      Q4 Coder ─┘
```

每个 Agent 各有一套独立的 system prompt 和历史记录，互不干扰。这里的关键是**并行执行 Coder**——两个模型同时写代码，相比串行调用节省了一半时间。

> **{think} 常见疑问：如果同一个模型串行三次，还算 Multi-Agent 吗？**
>
> 两个角度回答：
>
> **1. 关于"同一个模型"** — Agent 的定义是"承担一个角色的独立推理循环"，不是"不同模型"。三个 Agent 各有一套独立的 system prompt 和历史记录，相互看不到对方的上下文——它们"共享模型权重，不共享记忆"。所以即使底层是同一个模型，只要每个 Agent 有独立的角色定义和记忆空间，它就是 Multi-Agent，而不是同一个 Agent 调了三次。在实际项目中，你完全可以让 Coder 用 `gpt-4o`、Reviewer 用 `gpt-4o-mini`（省钱）、Judge 用 `claude-3-haiku`，只要换 `call_llm` 里的 model 参数就行。
>
> **2. 关于"串行"** — 这里 Coder 是并行的（`ThreadPoolExecutor`），Reviewer 和 Judge 因为有数据依赖才串行。多 Agent 不只有链式串行这一种模式：
>
| 模式 | 结构 | 适用场景 |
|------|------|---------|
| 链式串行 | A → B → C | Coder → Reviewer → Summarizer，前一步输出是后一步输入 |
| 并行投票（本例） | A、B 独立跑 → C 汇总 | 两个 Coder 各自写代码 → Reviewer 统一审查 |
| 辩论/讨论 | A ↔ B ↔ C 多轮对话 | 多个 Agent 扮演不同专家，互相挑战结论 |
| 编排调度 | Orchestrator 分配任务给 Worker | "经理 Agent"根据问题分派给不同的"专家 Agent" |
> 所以"多 Agent"的核心是**角色解耦 + 通信协议**，不是"多模型"也不是"非串行"。

以上三种方式都基于同一个底层逻辑：定义工具 schema → 传给模型 → 模型返回工具调用 → 执行工具 → 再次推理。无论你用的是 OpenAI、Ollama、还是本地 GGUF 模型，只需改模型加载部分，Agent 核心逻辑不变。

---

## 三种 Agent 写法对比

理解上面的底层逻辑后，你有三种路径可选。它们不是高低之分，是**适用场景不同**：

| | 自己写 Function Calling 循环 | LangChain Agent | AutoGen / 多Agent框架 |
|---|---|---|---|
| **本质** | 手动控制"思考→行动→观察"循环 | 框架帮你管理循环和记忆 | 多个Agent角色分工协作 |
| **适合场景** | 工具少（1-3个）、逻辑固定 | 工具中等（3-10个），需要记忆管理 | 复杂任务需要拆分给不同角色 |
| **你需要写的代码量** | 50-100行 | 20-30行 | 10-20行（但需要理解框架概念） |
| **灵活度** | 最高，你想怎么控制就怎么控制 | 中等，在框架抽象范围内自由 | 较低，受Agent协作模式的限制 |
| **调试难度** | 容易——东西都是你写的 | 中等——需要理解框架的ReAct执行逻辑 | 较难——多个Agent的对话流不好追踪 |

**怎么选**：

- 第一次接触 Agent → 先用 raw Function Calling 跑通一个 Demo（上面那段代码）。这一步不做就上框架，你会永远不理解"模型到底怎么调用工具的"。
- 做一个实际的 Agent 项目 → 选 LangChain Agent。它的 ReAct（Reasoning + Acting）模式是业界标准。定义一个搜索工具 + 一个计算器工具，让 Agent 回答"2024 年诺贝尔物理学奖得主的年龄是多少"——它会先搜谁获奖了，再算年龄，再回答。
- 任务需要多角色协作 → 用 [AutoGen](https://github.com/microsoft/autogen) 或 [MetaGPT](https://github.com/geekan/MetaGPT)。比如"写一个股票分析工具"——一个Agent写代码，一个Review代码，一个执行。分工比单Agent更可靠。

---

## 做深方向

**数据分析 Agent**：给一个CSV，让模型自动分析、画图、得出结论。

- 工具：[Open Interpreter](https://github.com/KillianLucas/open-interpreter)（本地代码执行，54k Star）或 [DB-GPT](https://github.com/eosphoros-ai/DB-GPT)（数据库+数据分析，13k Star）
- 怎么做：装好之后，给它一个 Excel，"统计上季度各门店的销售额趋势并画图"。第一次看到它自动写 Python 代码、执行、输出图表，你会意识到大量机械性数据分析工作可以被替代了。
- 简历话术："基于大模型的自动化数据分析系统，支持自然语言驱动的数据查询与可视化"

**信息聚合 Agent**：自动搜索多个来源，整合输出报告。

- 怎么做：定义三个工具——`search_web`、`fetch_page`、`summarize_text`。问 Agent "帮我调研一下目前最便宜的长上下文 LLM API"。它会依次搜索、打开页面、总结、整合。
- 简历话术："多工具协作的智能信息检索系统，支持自动搜索、内容提取与结构化报告生成"

**代码助手 Agent**：让 Agent 写代码、执行、纠错。

- 用 Open Interpreter 或自己搭一个代码沙箱。给 Agent 一个 Python 执行环境，让它写代码+运行，看结果是否符合预期。如果报错，让 Agent 自己修。

---

## 参考项目

| 项目 | 作用 |
|------|------|
| [openai-cookbook](https://github.com/openai/openai-cookbook) | Function Calling 的官方参考实现 |
| [litellm](https://github.com/BerriAI/litellm) | 统一多模型API，一行代码切换模型 |
| [LangChain](https://github.com/langchain-ai/langchain) | 生产级 Agent 框架，ReAct 模式标准实现 |
| [AutoGen](https://github.com/microsoft/autogen) | 微软多 Agent 框架，适合角色分工场景 |
| [Open Interpreter](https://github.com/KillianLucas/open-interpreter) | 数据分析 Agent 的快速启动项目 |

五个项目，从低到高覆盖了 Agent 的三个层次。

---

## 下篇预告

Agent 让你能用别人的模型做事情。但有一天你会想：如果我能让模型学会我自己领域的东西就好了——我们实验室的论文格式、我导师的写作风格、我们公司的产品知识。那你就需要下一篇的内容——微调（Fine-tuning）了。
---

> 本系列 Notebook 及配套文章：[github.com/ASPIRINH/hands-on-llm](https://github.com/ASPIRINH/hands-on-llm)

---

<details>
<summary><b>常见问题</b></summary>

**Q: `openai` 库调用 Ollama 报 `Connection refused`？**
A: Ollama 没有运行。先确认终端可以 `ollama run qwen2.5:7b` 正常对话。然后检查 `base_url` 是不是 `http://localhost:11434/v1`——注意是 `11434` 不是 `11433`，且不要忘了 `/v1`。

**Q: 模型没有返回 `tool_calls`，而是直接生成了一段文字回答？**
A: 你用的模型不支持 Function Calling。检查：1) `qwen2.5:7b` 的版本是否够新（`ollama pull qwen2.5:7b` 更新）；2) 测试用 `gpt-4o-mini` 如果接的是 OpenAI API；3) 换成支持 tool use 的模型如 `llama3.1:8b` 或 `mistral-nemo:12b`。

**Q: Function Calling 返回的 JSON 解析报错？**
A: 代码里 `tool_call.function.arguments` 是字符串（JSON），需要先 `json.loads()` 再取 key。改成：`args = json.loads(tool_call.function.arguments); query = args["query"]`。上面的代码示例为了简洁省略了这步。

**Q: LangChain Agent 的 ReAct 循环卡住了？**
A: 设置 `verbose=True` 可以看到每一步的思考过程。常见原因是工具返回的结果太长，超过了模型的上下文窗口。给工具加一个 `max_length` 参数截断返回内容。

**Q: Agent 反复调用同一个工具不停止？**
A: 加一个最大迭代次数限制。在 ReAct Agent 的配置里设 `max_iterations=5`，超过次数强制终止。这是 Agent 开发中最常见的 bug——没有上限的循环。

</details>
