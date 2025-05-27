# test_pagerank.py
import pytest
import re
from main import Graph, calPageRank  # 直接从main.py导入

@pytest.fixture
def test_graph():
    """构建包含测试数据的图对象"""
    graph = Graph()
    # 使用Easy Test.txt的内容构建图
    graph.build_from_text("Easy Test.txt")
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