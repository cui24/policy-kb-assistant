# Demo Data

This repository publishes exactly one bundled PDF for out-of-the-box demos:

- `data/demo/ACME_IT_Admin_Handbook_v1.0_Demo.pdf`

It is the default public sample document used by `python -m src.kb.ingest`.

## Why Only One Bundled File

The repository keeps third-party source PDFs out of version control to reduce copyright and redistribution risk.

That is why:
- `data/demo/` contains curated redistributable demo assets
- `data/raw/` is git-ignored and reserved for local-only documents

## Optional External Source Documents

If you want the extra public PDFs used during development, download them locally instead of committing them.

The repository includes:

- `scripts/download_pdfs.sh`

That script downloads these upstream files into `data/raw/`:

- `moe_student_management.pdf`
  - `https://www.moe.gov.cn/jyb_xxgk/xxgk/zhengce/guizhang/202112/P020211208551013106022.pdf`
- `pku_dorm_rules.pdf`
  - `https://yjsy.bjmu.edu.cn/docs/20210702153556447919.pdf`
- `henu_network_manual.pdf`
  - `https://ccce.henu.edu.cn/__local/E/6B/8E/3E047D0E443078120D8D2534818_33318A23_3AC6DC.pdf`

These links are provided for user-directed download only. They are not redistributed in this repository.

## How Ingestion Works

`python -m src.kb.ingest` scans both:

- `data/demo/*.pdf`
- `data/raw/*.pdf`

So the bundled ACME demo file works by default, and any locally downloaded PDFs are ingested automatically when present.
