#!/bin/bash

# 自动替换匿名链接脚本
# 用法：bash replace_links.sh <github_url> <zenodo_url>

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [ $# -ne 2 ]; then
    echo -e "${RED}错误：需要提供2个参数${NC}"
    echo ""
    echo "用法："
    echo "  bash replace_links.sh <GitHub_URL> <Zenodo_URL>"
    echo ""
    echo "示例："
    echo "  bash replace_links.sh \\"
    echo "    'https://github.com/anonymous-reviewer/antibody-cdr-control' \\"
    echo "    'https://zenodo.org/record/8234567?token=abcdefgh12345678'"
    echo ""
    exit 1
fi

GITHUB_URL="$1"
ZENODO_URL="$2"

echo -e "${GREEN}=== 匿名链接替换脚本 ===${NC}"
echo ""
echo "GitHub URL: $GITHUB_URL"
echo "Zenodo URL: $ZENODO_URL"
echo ""

# 备份原始文件
echo -e "${YELLOW}步骤1：备份原始文件...${NC}"
cp manuscript/main_manuscript.md manuscript/main_manuscript.md.backup
cp manuscript/cover_letter.md manuscript/cover_letter.md.backup
echo "✓ 备份完成"
echo ""

# 替换main_manuscript.md中的链接
echo -e "${YELLOW}步骤2：替换main_manuscript.md中的链接...${NC}"

# 替换GitHub链接（line 639）
sed -i "s|\[ANONYMOUS_REPO_LINK - to be provided at submission\]|$GITHUB_URL|g" manuscript/main_manuscript.md

# 替换Zenodo链接（line 640）
sed -i "s|\[ANONYMOUS_ZENODO_LINK - to be provided at submission\]|$ZENODO_URL|g" manuscript/main_manuscript.md

echo "✓ main_manuscript.md 替换完成"
echo ""

# 替换cover_letter.md中的链接
echo -e "${YELLOW}步骤3：替换cover_letter.md中的链接...${NC}"

sed -i "s|\[ANONYMOUS_REPO_LINK - to be provided at submission\]|$GITHUB_URL|g" manuscript/cover_letter.md
sed -i "s|\[ANONYMOUS_ZENODO_LINK - to be provided at submission\]|$ZENODO_URL|g" manuscript/cover_letter.md

echo "✓ cover_letter.md 替换完成"
echo ""

# 验证替换结果
echo -e "${YELLOW}步骤4：验证替换结果...${NC}"

# 检查是否还有残留占位符
REMAINING=$(grep -n "ANONYMOUS_REPO_LINK\|ANONYMOUS_ZENODO_LINK" manuscript/*.md || true)

if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}✓ 验证通过：所有占位符已成功替换${NC}"
    echo ""
    echo "替换后的链接："
    echo ""
    echo "main_manuscript.md:"
    grep -n "Code repository\|Model checkpoints" manuscript/main_manuscript.md | head -2
    echo ""
    echo "cover_letter.md:"
    grep -n "Code repository\|Model checkpoints" manuscript/cover_letter.md | head -2
    echo ""
    echo -e "${GREEN}=== 替换完成！可以投稿了 ===${NC}"
    echo ""
    echo "备份文件位置："
    echo "  - manuscript/main_manuscript.md.backup"
    echo "  - manuscript/cover_letter.md.backup"
    echo ""
    echo "如需恢复原始文件："
    echo "  mv manuscript/main_manuscript.md.backup manuscript/main_manuscript.md"
    echo "  mv manuscript/cover_letter.md.backup manuscript/cover_letter.md"
else
    echo -e "${RED}✗ 警告：发现残留占位符${NC}"
    echo "$REMAINING"
    echo ""
    echo "请手动检查并修复。"
    exit 1
fi
