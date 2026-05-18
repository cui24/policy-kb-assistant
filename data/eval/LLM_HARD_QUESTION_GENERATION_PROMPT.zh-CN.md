# Hard 评测题生成提示词（v1）

你是企业制度知识库评测集标注员。请基于给定制度片段，生成**仅 hard 难度**问题。

## 目标
生成 3 类高区分问题：
1. `flow_cross_segment`（跨段流程题）
2. `condition_exception`（条件 + 例外题）
3. `similar_policy_interference`（相似制度干扰题）

## 输入
你会收到一个 JSON，包含：
- `category`：题目所属类别（hr/admin/finance/it）
- `tasks`：题目骨架（每题含 `query_id/category/difficulty/query_type/class_tag/recommended_chunk_ids`）
- `chunks`：证据片段池（`doc_id/chunk_id/section_path/text/rule_type/evidence_span/keywords`）
- `documents`：文档元数据（`doc_id/effective_date/status/version/family_id/authority_level`）

## 输出格式（严格）
仅输出一个 JSON 数组，不要输出额外说明。每个元素必须包含：
- `query_id`
- `query`
- `category`
- `difficulty`
- `query_type`
- `gold_doc_id`
- `gold_section`
- `answer_points`
- `as_of_date`
- `conflict_case`
- `notes`

示例：
```json
{
  "query_id": "QH_FIN_001",
  "query": "员工因公借支后到报销完成需要经过哪些步骤？若超过规定时限应如何处理？",
  "category": "finance",
  "difficulty": "hard",
  "query_type": "multi_hop",
  "gold_doc_id": "FIN-05",
  "gold_section": "第三章 资金管理 > 第六条 现金的管理 > 6.3 个人因公借款管理",
  "answer_points": "备用金须由部门申请并经副总经理批准；业务完成后一周内持审批单据与出纳结算缴销；所有借支需统一填制专用借款借据；办理还款时由出纳开具财务专用章收款收据",
  "as_of_date": "",
  "conflict_case": "no",
  "notes": ""
}
```

## 强约束
1. `difficulty` 必须为 `hard`。
2. `query_type`、`category` 不得改动，必须与任务骨架一致。
3. `query` 必须是自然用户问法，不能出现文档编号（如 `HR-01`、`FIN-06`）、“第X条”等泄题表达。
4. `gold_doc_id` 必须来自输入文档集合。
5. `gold_section` 必须能在输入 `chunks.section_path` 中找到可追溯依据。
6. `answer_points` 必须 3-6 条事实点，用中文分号 `；` 分隔；每一点都能从证据定位。
7. `conflict_case` 默认 `no`；仅当确有版本/时点冲突才可填 `yes`。
8. 当 `conflict_case=yes` 时，`as_of_date` 必须是 `YYYY-MM-DD`，并在 `notes` 写清冲突来源。
9. 不得编造制度中不存在的金额、流程、时限、角色、处罚。
10. 输出条目数量必须与输入 tasks 一致，不得缺题、不得重复 query_id。

## 按题型生成要求
### A. `flow_cross_segment`
- 问题必须要求“完整链路”或“步骤 + 时限/角色”联合回答。
- 优先使用 `recommended_chunk_ids` 中多个片段，避免只问单条定义。

### B. `condition_exception`
- 问题必须包含至少一个触发条件与一个例外/禁止边界。
- 禁止把“例外”写成虚构场景，必须来自证据文本。

### C. `similar_policy_interference`
- 问题必须体现两个相似制度概念的区分（职责、条件、标准、流程差异）。
- 避免仅关键词同义改写，必须有可核验“差异点”。

## 生成前自检
1. 每题都能被 `gold_doc_id + gold_section` 支撑。
2. `answer_points` 能一一映射到证据。
3. `query` 无文档编号、无条款号泄露。
4. 保持 hard 难度，不退化成简单事实题。

请根据接下来提供的输入数据输出最终 JSON 数组。
