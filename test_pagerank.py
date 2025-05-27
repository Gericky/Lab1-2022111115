# test_pagerank.py
import pytest
import re
from main import Graph, calPageRank  # 直接从main.py导入

@pytest.fixture
def test_graph():
    """使用Easy Test.txt构建测试图"""
    graph = Graph()
    # 硬编码构建过程以保证测试可靠性
    test_text = """
    the scientist carefully analyzed the data 
    wrote a detailed report and shared the report 
    with the team but the team requested more data 
    so the scientist analyzed it again
    """
    words = re.findall(r'\b[a-z]+\b', test_text.lower())
    for i in range(len(words)-1):
        graph.add_edge(words[i], words[i+1])
    # 手动设置词频计数器
    graph.word_counts = {
        "the": 6, "scientist": 2, "carefully": 1,
        "analyzed": 2, "data": 2, "wrote": 1,
        "a": 1, "detailed": 1, "report": 2,
        "and": 1, "shared": 1, "with": 1,
        "team": 2, "but": 1, "requested": 1,
        "more": 1, "so": 1, "it": 1, "again": 1
    }
    return graph

# 测试用例1：Data (覆盖等价类1,2)
def test_case_1(test_graph):
    result = calPageRank(test_graph, "data")
    assert pytest.approx(result, abs=0.001) == 0.076, "测试用例1失败"

# 测试用例2：Carefully (覆盖等价类1)
def test_case_2(test_graph):
    result = calPageRank(test_graph, "carefully")
    assert pytest.approx(result, abs=0.001) == 0.029, "测试用例2失败"

# 测试用例3：The (覆盖等价类1,2)
def test_case_3(test_graph):
    result = calPageRank(test_graph, "the")
    assert pytest.approx(result, abs=0.001) == 0.175, "测试用例3失败"

# 测试用例4：Noon (覆盖等价类4)
def test_case_4(test_graph):
    result = calPageRank(test_graph, "noon")
    assert result == 'No "noon" in the graph!', "测试用例4失败"