#!/bin/bash

# 🚀 Hugging Face Spaces 快速部署脚本
# Usage: ./deploy_to_hf.sh YOUR_HF_USERNAME YOUR_SPACE_NAME

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [ "$#" -ne 2 ]; then
    echo -e "${RED}错误: 需要提供 HF 用户名和 Space 名称${NC}"
    echo "用法: ./deploy_to_hf.sh YOUR_HF_USERNAME YOUR_SPACE_NAME"
    echo "示例: ./deploy_to_hf.sh john-doe industrial-twin"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 Hugging Face Spaces 部署工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "目标 Space: ${YELLOW}${SPACE_URL}${NC}"
echo ""

# 步骤 1: 检查 Git LFS
echo -e "${YELLOW}步骤 1/6: 检查 Git LFS...${NC}"
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}❌ Git LFS 未安装${NC}"
    echo "请先安装 Git LFS:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  macOS: brew install git-lfs"
    echo "  Windows: https://git-lfs.github.com/"
    exit 1
fi
git lfs install
echo -e "${GREEN}✅ Git LFS 已安装${NC}"
echo ""

# 步骤 2: 创建临时部署目录
echo -e "${YELLOW}步骤 2/6: 准备部署文件...${NC}"
DEPLOY_DIR="hf_deploy_temp"
rm -rf ${DEPLOY_DIR}
mkdir -p ${DEPLOY_DIR}

# 步骤 3: 复制必要文件
echo -e "${YELLOW}步骤 3/6: 复制项目文件...${NC}"

# 核心文件
cp app.py ${DEPLOY_DIR}/
cp gradio_sensor_transformer_app.py ${DEPLOY_DIR}/
cp .gitattributes ${DEPLOY_DIR}/

# 依赖和配置
cp requirements_hf.txt ${DEPLOY_DIR}/requirements.txt
cp README_HF_SPACES.md ${DEPLOY_DIR}/README.md

# 代码目录
cp -r models ${DEPLOY_DIR}/
cp -r src ${DEPLOY_DIR}/

# 可选：模型和数据
if [ -d "saved_models" ] && [ "$(ls -A saved_models)" ]; then
    echo -e "${GREEN}  → 发现预训练模型，正在复制...${NC}"
    cp -r saved_models ${DEPLOY_DIR}/
else
    echo -e "${YELLOW}  ⚠️  未发现预训练模型（可稍后上传）${NC}"
    mkdir -p ${DEPLOY_DIR}/saved_models
fi

if [ -d "data" ] && [ "$(ls -A data)" ]; then
    echo -e "${GREEN}  → 发现示例数据，正在复制...${NC}"
    cp -r data ${DEPLOY_DIR}/
else
    echo -e "${YELLOW}  ⚠️  未发现示例数据（可稍后上传）${NC}"
    mkdir -p ${DEPLOY_DIR}/data
fi

echo -e "${GREEN}✅ 文件复制完成${NC}"
echo ""

# 步骤 4: 克隆 HF Space（如果还没有）
echo -e "${YELLOW}步骤 4/6: 连接到 Hugging Face Space...${NC}"
echo -e "${GREEN}请按照提示操作：${NC}"
echo -e "  1. 如果 Space 不存在，请先在 HF 网站创建："
echo -e "     ${YELLOW}https://huggingface.co/new-space${NC}"
echo -e "  2. 确保选择 SDK: ${GREEN}Gradio${NC}"
echo -e "  3. 克隆仓库到 ${DEPLOY_DIR} 目录"
echo ""
echo -e "${YELLOW}是否继续克隆 Space 仓库? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    cd ${DEPLOY_DIR}
    git clone ${SPACE_URL} temp_repo

    # 将仓库内容移到当前目录
    if [ -d "temp_repo/.git" ]; then
        mv temp_repo/.git ./
        rm -rf temp_repo
        echo -e "${GREEN}✅ Space 仓库已克隆${NC}"
    else
        echo -e "${RED}❌ 克隆失败，请检查 Space 是否存在${NC}"
        exit 1
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  跳过克隆步骤${NC}"
    echo -e "请手动克隆后运行部署命令"
    exit 0
fi
echo ""

# 步骤 5: Git 提交
echo -e "${YELLOW}步骤 5/6: 提交更改...${NC}"
cd ${DEPLOY_DIR}

git add .
git commit -m "Deploy Industrial Digital Twin application

- Add Gradio interface
- Include model architecture code
- Set up directory structure
" || echo "没有需要提交的更改"

echo -e "${GREEN}✅ 更改已提交${NC}"
echo ""

# 步骤 6: 推送到 HF
echo -e "${YELLOW}步骤 6/6: 推送到 Hugging Face...${NC}"
echo -e "${GREEN}请输入您的 HF Access Token${NC}"
echo -e "（获取: https://huggingface.co/settings/tokens）"
echo ""

git push

cd ..

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "您的应用正在构建中，请访问："
echo -e "${YELLOW}${SPACE_URL}${NC}"
echo ""
echo -e "构建通常需要 3-5 分钟"
echo -e "您可以在 Space 页面查看构建日志"
echo ""
echo -e "清理临时文件: ${YELLOW}rm -rf ${DEPLOY_DIR}${NC}"
echo ""
