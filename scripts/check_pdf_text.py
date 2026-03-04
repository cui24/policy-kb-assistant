"""
PDF 文本抽样验收程序：快速判断原始 PDF 是否适合进入 L0 入库流程。

一、程序目标
1. 读取指定目录下的 PDF 文件。
2. 对每个 PDF 抽样检查若干页面是否能提取出文本。
3. 若文本明显为空或异常过短，则判定存在风险。
4. 用退出码告诉调用方是否通过验收。

二、程序入口与运行顺序
1. 命令入口：`python scripts/check_pdf_text.py <pdf_dir>`
2. Python 入口：模块底部执行 `main()`
3. `main()` 内部顺序如下：
   3.1 读取命令行参数，拿到 PDF 目录
   3.2 扫描目录中的 `*.pdf`
   3.3 逐个打开 PDF
   3.4 抽样检查前两页；若页数足够，再检查中间页
   3.5 对每个抽样页打印字符数和样本文本
   3.6 如果存在异常，则把 `ok` 置为 `False`
   3.7 最后根据 `ok` 输出 PASS/FAIL，并用退出码结束

三、主要函数的输入输出
1. `sample_text(s: str, n: int = 240) -> str`
   - 输入：原始文本、截断长度
   - 输出：压缩空白后的短样本字符串

2. `main() -> None`
   - 输入：依赖命令行参数 `sys.argv[1]`
   - 输出：无返回值
   - 副作用：
     - 读取 PDF
     - 向终端打印抽样结果
     - 通过 `sys.exit(code)` 返回状态码

四、退出码约定
1. `0`：检查通过
2. `1`：参数错误或目录下无 PDF
3. `2`：发现空白/异常 PDF，不建议直接入库

五、程序可以理解成的伪代码
1. 读取目录参数
2. 找到目录中的所有 PDF
3. 如果没有 PDF，直接退出
4. 对每个 PDF：
   4.1 打开文件
   4.2 决定抽样页
   4.3 提取文本并打印样本
   4.4 如果文本太短，标记失败
5. 根据整体结果输出 PASS 或 FAIL
6. 返回对应退出码
"""

import sys
from pathlib import Path
from pypdf import PdfReader

def sample_text(s: str, n=240):
    s = " ".join(s.split())
    return s[:n] + ("..." if len(s) > n else "")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_pdf_text.py <pdf_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    pdfs = sorted(root.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {root}")
        sys.exit(1)

    ok = True
    for p in pdfs:
        print("=" * 90)
        print(f"[FILE] {p.name}")
        try:
            r = PdfReader(str(p))
            n_pages = len(r.pages)
            print(f"[PAGES] {n_pages}")
            """抽样检查前两页；如果页数足够，再额外检查中间页。"""
            idxs = [0, 1] if n_pages >= 2 else [0]
            if n_pages >= 5:
                idxs.append(n_pages // 2)
            for i in idxs:
                txt = (r.pages[i].extract_text() or "")
                print(f"[PAGE {i+1}] chars={len(txt)} sample={sample_text(txt)}")
                if len(txt.strip()) < 50:
                    ok = False
        except Exception as e:
            ok = False
            print(f"[ERROR] {e}")

    print("=" * 90)
    if ok:
        print("[PASS] PDFs look extractable (non-empty, non-garbled samples).")
        sys.exit(0)
    else:
        print("[FAIL] Some PDFs appear empty/garbled. Replace those PDFs (L0 does NOT OCR).")
        sys.exit(2)

if __name__ == "__main__":
    main()
