#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw

curl -L "https://www.moe.gov.cn/jyb_xxgk/xxgk/zhengce/guizhang/202112/P020211208551013106022.pdf" \
  -o data/raw/moe_student_management.pdf

curl -L "https://yjsy.bjmu.edu.cn/docs/20210702153556447919.pdf" \
  -o data/raw/pku_dorm_rules.pdf

curl -L "https://ccce.henu.edu.cn/__local/E/6B/8E/3E047D0E443078120D8D2534818_33318A23_3AC6DC.pdf" \
  -o data/raw/henu_network_manual.pdf
