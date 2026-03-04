"""
L1 命令行演示程序：把检索与回答串成一个可直接展示的入口。

一、程序目标
1. 接收命令行问题或交互输入问题。
2. 调用 `retrieve(...)` 获取证据。
3. 调用 `answer_with_citations(...)` 生成最终答案。
4. 以人可读的终端格式打印答案和引用。

二、程序入口与运行顺序
1. 命令入口：`python -m src.cli.demo_cli "问题"`
2. Python 入口：模块底部执行 `main()`
3. `main()` 内部顺序如下：
   3.1 `load_dotenv()`：加载环境变量
   3.2 解析命令行参数
   3.3 若命令行没有问题，则进入交互输入
   3.4 调用 `retrieve(question, top_k=...)`
   3.5 调用 `answer_with_citations(question, hits)`
   3.6 打印 `ANSWER`
   3.7 打印 `CITATIONS`

三、主要函数的输入输出
1. `main() -> None`
   - 输入：无显式参数，依赖命令行参数或终端输入
   - 输出：无返回值
   - 副作用：向终端打印内容

四、关键数据流
1. 输入问题：`str`
2. `retrieve(...)` 输出：`list[dict]`
3. `answer_with_citations(...)` 输出：
   {
     "answer": str,
     "citations": list[{
       "doc_id": str,
       "page": int,
       "snippet": str
     }]
   }
4. 最终打印格式：
   - `=== ANSWER ===`
   - 答案正文
   - `=== CITATIONS ===`
   - 每条引用以 `- doc_id p页码: snippet` 展示

五、程序可以理解成的伪代码
1. 读环境变量
2. 读命令行参数
3. 若没传问题则从终端读取
4. 调检索
5. 调回答
6. 把结果打印给用户
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.kb.answer import answer_with_citations
from src.kb.retrieve import retrieve


def main() -> None:
    """这里也加载 .env，是为了让 CLI 直接继承和主程序一致的配置。"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="政策知识库命令行演示入口")
    parser.add_argument("question", nargs="*", help="问题文本")
    parser.add_argument("--topk", type=int, default=None, help="临时覆盖检索 top-k")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        """允许不带参数进入交互模式，便于面试时现场手输问题。"""
        question = input("Question> ").strip()
    if not question:
        raise SystemExit("Question is empty.")

    """CLI 不做额外业务逻辑，只负责串起 retrieve -> answer，保持入口足够薄。"""
    hits = retrieve(question, top_k=args.topk)
    result = answer_with_citations(question, hits)

    print("\n=== ANSWER ===")
    print(result.get("answer", ""))

    print("\n=== CITATIONS ===")
    citations = result.get("citations", []) or []
    if not citations:
        print("(none)")
        return
    """引用只展示前 10 条，避免终端输出过长影响可读性。"""
    for item in citations[:10]:
        doc_id = item.get("doc_id")
        page = item.get("page")
        snippet = str(item.get("snippet", ""))[:160]
        print(f"- {doc_id} p{page}: {snippet}")


if __name__ == "__main__":
    main()
