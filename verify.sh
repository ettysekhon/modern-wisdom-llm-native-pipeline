#!/bin/bash
set -e

echo "🔍 Verifying migration..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

test_step() {
    local name=$1
    local command=$2
    
    echo -n "Testing $name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC}"
        ((FAILED++))
        return 1
    fi
}

# 1. Test imports
test_step "Python imports" "uv run python -c 'from modern_wisdom_rag_pipeline import paths; from modern_wisdom_rag_pipeline.agent import run_deep_agent; from modern_wisdom_rag_pipeline.tracing import get_tracer'"

# 2. Test CLI
test_step "CLI installation" "uv run mw-rag --help"

# 3. Test Qdrant connection (if running)
if docker compose ps qdrant 2>/dev/null | grep -q "Up"; then
    test_step "Qdrant connection" "uv run mw-rag list"
else
    echo -e "Skipping Qdrant test (not running) ${YELLOW}⚠${NC}"
fi

# 4. Test tracing initialization
test_step "Tracing initialization" "uv run python -c 'from modern_wisdom_rag_pipeline.tracing import get_tracer; get_tracer()'"

# 5. Test agent (mock mode)
test_step "Agent execution (mock)" "uv run python -c '
from modern_wisdom_rag_pipeline.agent import run_deep_agent
result = run_deep_agent(\"test\", \"\", 5, \"corpus\", \"mock\", \"mock\", 2)
assert \"result\" in result
'"

# 6. Check for spike imports in app code
if grep -r "from spike" src/modern_wisdom_rag_pipeline/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__" > /dev/null; then
    echo -e "Checking spike imports... ${RED}✗ Found spike imports in app code${NC}"
    ((FAILED++))
else
    echo -e "Checking spike imports... ${GREEN}✓ No spike imports found${NC}"
    ((PASSED++))
fi

# 7. Test docker compose config
if docker compose config > /dev/null 2>&1; then
    echo -e "Docker Compose config... ${GREEN}✓${NC}"
    ((PASSED++))
else
    echo -e "Docker Compose config... ${RED}✗${NC}"
    ((FAILED++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All basic checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Start services: docker compose up -d"
    echo "  2. Load embeddings: uv run mw-rag upsert-batch --episode-list data/tmp/epids_2018_2025.txt --emb-v 'BAAI/bge-small-en-v1.5' --set-live"
    echo "  3. Test in browser: http://localhost:8001"
    echo "  4. Check traces: http://localhost:6006"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. See VERIFICATION.md for detailed steps.${NC}"
    exit 1
fi

