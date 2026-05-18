# 评测集构建最短执行清单（MVP）

## 0. 单一事实源
- 文档主表只维护这一份：`data/eval/documents.csv`
- 当前固定列：
`doc_id,title,category,policy_type,doc_family,source_file,status,version,effective_date,family_id,authority_level,notes`

## 1. 先规范化并校验 documents.csv
```bash
python scripts/normalize_documents_csv.py
python scripts/validate_documents_csv.py
```

说明：
- `status` 支持：`active / archived / unknown`
- `effective_date` 可空；有值时必须 `YYYY-MM-DD`
- `family_id` 必填（脚本会自动补默认值）

## 2. 生成 100 题模板（4 类各 25）
```bash
python scripts/generate_rag_eval_template.py
python scripts/validate_rag_eval_set.py
```

输出文件：
- `data/eval/rag_eval_set_v2_100_template.csv`

## 3. 四窗口让 LLM 生成题（你手动执行）
- finance 窗口生成 `Q_FIN_001..Q_FIN_025`
- hr 窗口生成 `Q_HR_001..Q_HR_025`
- admin 窗口生成 `Q_ADM_001..Q_ADM_025`
- it 窗口生成 `Q_IT_001..Q_IT_025`

建议每个窗口输出单独 CSV：
- `data/eval/finance_eval.csv`
- `data/eval/hr_eval.csv`
- `data/eval/admin_eval.csv`
- `data/eval/it_eval.csv`

## 4. 回填并校验总表
你将四个窗口结果合并进：
- `data/eval/rag_eval_set_v2_100_template.csv`

然后校验：
```bash
python scripts/validate_rag_eval_set.py
```

## 5. 人工证据抽查（最低标准）
- 每类至少抽查 5 条，共 20 条
- 必查项：
  - `gold_doc_id` 是否准确
  - `gold_section` 是否可定位
  - `answer_points` 是否可由证据支持
  - `conflict_case=yes` 是否真的存在冲突且 `as_of_date` 有值

## 6. 冻结正式评测集
抽查通过后复制为正式文件：
```bash
cp data/eval/rag_eval_set_v2_100_template.csv data/eval/policy_eval_100.csv
```

正式评测使用：
- `data/eval/policy_eval_100.csv`

## 7. 入库与评测（下一阶段）
前置满足后再做：
- 三套切块策略入库（baseline / overlap / structured）
- 同一 `policy_eval_100.csv` 跑 `Recall@3/5, MRR, GoldDoc Hit@5`
