import pytest
from main import Graph, queryBridgeWords

@pytest.fixture
def test_graph():
    """构建包含测试数据的图对象"""
    graph = Graph()
    # 使用Easy Test.txt的内容构建图
    graph.build_from_text("Easy Test.txt")
    return graph



@pytest.mark.parametrize("input_words, expected_output", [
    # 测试用例1（word1存在，word2存在，无桥接词）
    (("the", "report"), 'No bridge words from "the" to "report"!'),

    # 测试用例2（存在单桥接词）
    (("scientist", "the"), 'The bridge word from "scientist" to "the" is: analyzed'),

    # 测试用例3（word2不存在）
    (("the", "x"), 'No "x" in the graph!'),

    # 测试用例4（word1不存在）
    (("x", "the"), 'No "x" in the graph!')
])
def test_bridge_words(test_graph, input_words, expected_output):
    """测试桥接词查询功能"""
    word1, word2 = input_words
    # 执行查询（自动处理大小写）
    result = queryBridgeWords(test_graph, word1.lower(), word2.lower())

    # 精确匹配输出内容（包括标点和大小写）
    assert result == expected_output, \
        f"测试失败：输入({word1}, {word2}) 期望：{expected_output} 实际：{result}"


