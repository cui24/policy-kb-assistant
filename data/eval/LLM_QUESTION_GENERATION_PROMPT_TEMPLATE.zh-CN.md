# 企业政策库评测题生成提示词（v1）

你是企业制度评测集标注员。请基于给定的制度片段，生成**高质量、可核验**的评测问题，并输出为结构化 JSON。

## 输入
你会收到：
1. 题目骨架列表（来自 `rag_eval_set_v2_100_template.csv`，每行含 `query_id/category/difficulty/query_type`）
2. 制度片段（来自 `data/extracted/**.jsonl`，字段含 `doc_id/section_path/text/rule_type/evidence_span`）
3. 文档元数据（来自 `data/eval/documents.csv`，含 `doc_id/effective_date/status/version/family_id/authority_level`）

## 任务
对每个输入 `query_id` 生成以下字段：
- `query`
- `gold_doc_id`
- `gold_section`
- `answer_points`
- `as_of_date`
- `conflict_case`
- `notes`

并保留原有：
- `query_id`
- `category`
- `difficulty`
- `query_type`

## 输出格式（严格）
仅输出一个 JSON 数组，不要输出任何解释文本。数组元素结构如下：
```json
{
  "query_id": "Q_HR_001",
  "query": "员工完成出差任务后，最晚应在多久内完成报销？需附什么材料？",
  "category": "hr",
  "difficulty": "easy",
  "query_type": "procedure",
  "gold_doc_id": "FIN-02",
  "gold_section": "第三章 因公出差的借款及报销 > 第十条",
  "answer_points": "完成出差后一周内报销；填写《差旅费报销单》；按规定审批；需附《出差申请书》",
  "as_of_date": "",
  "conflict_case": "no",
  "notes": ""
}
```

## 强约束
1. `gold_doc_id` 必须来自给定文档集合，且应是最权威单一依据（单值）。
2. `gold_section` 必须可追溯到输入片段中的 `section_path`。
3. `answer_points` 必须是可核验事实点，3-6 个，用中文分号 `；` 分隔，不写空话。
4. `query` 必须是自然用户问法，避免直接抄条文原句。
5. `query` 与 `difficulty/query_type` 一致：
   - `easy`：单条规则直接问；
   - `medium`：带条件或两步判断；
   - `hard`：跨条款整合、边界条件、例外处理。
6. `query_type` 语义：
   - `fact`：定义/标准/额度/时限
   - `procedure`：流程/审批链路/时序
   - `constraint`：禁止/必须/责任追究
   - `comparison`：不同级别/场景差异
   - `exception`：例外条件/豁免
   - `multi_hop`：跨段组合推理
7. `conflict_case` 默认 `no`；仅当确有版本冲突或时点冲突时填 `yes`。
8. 若 `conflict_case=yes`，必须填写：
   - `as_of_date`（`YYYY-MM-DD`）
   - `notes`（冲突来源，如 `FIN-02 v1.0 vs FIN-02 v2.0`）
9. 不得编造制度中不存在的金额、时限、角色、处罚。
10. 输出条目数量必须与输入骨架数量一致，`query_id` 不得重复或缺失。

## 质量自检（生成前先内部检查）
1. 每题都能被 `gold_doc_id + gold_section` 唯一支持。
2. `answer_points` 每一点都能在片段中找到证据。
3. `query` 不含“根据第几条”这类泄题表达。
4. 分类一致：`category` 与问题主题一致（hr/admin/finance/it）。
5. 冲突题占比控制在 10%-15%（若输入批次样本量允许）。

## 现在开始
请根据我接下来提供的输入数据，输出最终 JSON 数组。

