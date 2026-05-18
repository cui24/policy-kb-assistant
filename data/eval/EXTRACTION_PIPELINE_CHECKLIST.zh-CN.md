# 结构化片段抽取执行清单（LLM版）

## 1. 准备输入
1. 文档元数据：`data/eval/prompt_inputs/*_documents.csv`
2. 快速文本参考：`data/eval/prompt_inputs/*_prompt_source.csv`
3. 抽取提示词模板：`data/eval/LLM_CHUNK_EXTRACTION_PROMPT_TEMPLATE.zh-CN.md`

## 2. 让 LLM 按“单文档”输出 JSONL
- 每个文档生成一个文件：
  - `data/extracted/finance/FIN-01.jsonl`
  - `data/extracted/hr/HR-01.jsonl`
  - `data/extracted/admin/ADM-01.jsonl`
  - `data/extracted/it/IT-01.jsonl`

## 3. 校验抽取结果（严格）
```bash
python scripts/validate_extracted_chunks.py \
  --input data/extracted \
  --documents data/eval/documents.csv \
  --quickscan data/eval/document_quick_scan.csv \
  --out-csv data/extracted/all_chunks_validated.csv
```

## 4. 快速合并（非严格）
```bash
python scripts/merge_extracted_chunks.py \
  --input-dir data/extracted \
  --out-csv data/extracted/all_chunks.csv
```

## 5. 人工抽查建议
- 每个类别至少抽查 3 个 chunk：
  - `section_path` 是否可定位
  - `evidence_span` 是否忠实原文
  - `rule_type` 是否合理
  - `effective_date` 是否真实或留空
