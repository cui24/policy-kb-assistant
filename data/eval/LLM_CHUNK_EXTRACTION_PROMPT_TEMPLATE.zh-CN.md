# LLM 结构化片段抽取提示词模板（按单个文档执行）

> 注释（调用建议）  
> 1. 一次只喂一个文档，避免上下文混文档导致 `doc_id/chunk_id` 污染。  
> 2. 输入分页文本时，保留 page 标记，便于模型正确填写 `page_start/page_end`。  
> 3. 输出必须是 JSONL 纯文本，不要 code fence，不要解释文字。  
> 4. 输出后必须跑 `scripts/validate_extracted_chunks.py` 做结构与证据一致性校验。  
> 5. 若校验失败，优先修 `keywords`（必须在 text 中可检索）与 `evidence_span`（必须是连续子串）。

你是企业制度知识库的数据标注助手。  
请你**仅基于输入文本**抽取可追溯的结构化片段，不得编造。

## 输入
我会给你：
1. 文档元数据（doc_id/title/category/policy_type/doc_family/status/version/effective_date/family_id/source_file）
2. 文档分页文本（每段带 page）

## 任务
请将文档抽取为 JSONL（每行一个 JSON 对象），字段必须严格包含：
- `doc_id`：字符串，必须与输入一致
- `chunk_id`：字符串，建议格式 `DOCID_chunk_0001`
- `section_path`：字符串，章节路径，尽量具体到“章/节/条”
- `page_start`：整数
- `page_end`：整数（`page_end >= page_start`）
- `text`：字符串，原文片段（不要改写）
- `keywords`：字符串数组，3-8 个关键词
- `rule_type`：枚举之一：`fact/procedure/constraint/comparison/exception/multi_hop/definition/checklist`
- `effective_date`：字符串，格式 `YYYY-MM-DD`，无明确日期填空字符串 `""`
- `evidence_span`：字符串，直接证据短句（来自原文）

## 抽取要求（强约束）
1. 只允许“原文摘录”，`text` 不得改写、扩写、总结、拼接外部内容。
2. 不要输出与制度无关的页眉页脚、目录噪声。
3. 每个 chunk 只允许一个主规则意图（definition/procedure/constraint 等），不要在一个 chunk 混多个主意图。
4. 每个 chunk 建议 150-500 字；超过 500 字必须拆分。
5. 如果同一条款跨页，允许 `page_start != page_end`。
6. `section_path` 尽量沿用文档标题层级；无法判断时写最接近的标题。
7. `keywords` 必须全部能在 `text` 中找到对应词（或同形短语），禁止“猜测关键词”。
8. `evidence_span` 必须是 `text` 的连续子串，不得改写。
9. `rule_type` 选择最贴近该片段主功能的类型，优先规则：
   - 定义性条款 -> `definition`
   - 操作步骤/审批流程 -> `procedure`
   - 禁止、限制、责任、处罚、阈值 -> `constraint` 或 `exception`
10. 不要输出 Markdown，不要解释，只输出 JSONL。

## 自检后再输出
在最终输出前逐条自检：
1. `text` 是否原文摘录（无改写）？
2. `keywords` 是否都能在 `text` 找到？
3. `evidence_span` 是否为 `text` 连续子串？
4. 单条是否超过 500 字并应拆分？
5. `rule_type` 是否匹配主意图？
6. keywords 只能用原文里确实出现的词。
7. evidence_span 复制 text 中一段连续原句，不加省略号。（evidence_span是从 text 里截的一小段关键证据句（短引用，用来快速核对））

## 输出示例（仅示例，不可抄内容）
{"doc_id":"FIN-02","chunk_id":"FIN-02_chunk_0001","section_path":"第二章 因公出差的审批 > 第五条","page_start":2,"page_end":2,"text":"因工作需要出差，由申请人填写《出差申请书》并提交审批。","keywords":["出差","审批","申请书"],"rule_type":"procedure","effective_date":"","evidence_span":"因工作需要出差，由申请人填写《出差申请书》并提交审批。"}
{"doc_id":"FIN-02","chunk_id":"FIN-02_chunk_0002","section_path":"第二章 因公出差的审批 > 第六条","page_start":2,"page_end":2,"text":"一般员工因公出差，由申请人部门负责人审核、行政部负责人审核，财务部门负责人合议、副总经理批准。","keywords":["审核","审批层级","副总经理"],"rule_type":"constraint","effective_date":"","evidence_span":"一般员工因公出差，由申请人部门负责人审核...副总经理批准。"}
