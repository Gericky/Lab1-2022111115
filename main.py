import re
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Graph:
    def __init__(self):
        self.vertices= []
        self.edges= defaultdict(list)
        self.weights={}
    def add_vertex(self,vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)

    def add_edge(self,from_vertex,to_vertex):
        if from_vertex not in self.vertices:
            self.add_vertex(from_vertex)
        if to_vertex not in self.vertices:
            self.add_vertex(to_vertex)
        self.edges[from_vertex].append(to_vertex)
        edge=(from_vertex,to_vertex)
        self.weights[edge]=self.weights.get(edge,0)+1

    def build_from_text(self,filename):
        try:
            with open(filename,'r',encoding='utf-8') as file:
                text = file.read().lower()
                word = re.findall(r'\b[a-z]+\b',text)
                for i in range(len(word)-1):
                    self.add_edge(word[i],word[i+1])
                return True
        except Exception :
            return False

def showDirectedGraph(graph):
    result="有向图结构如下：\n"
    result+=f"顶点数：{len(graph.vertices)}\n"
    result+=f"边数：{sum(len(v) for v in graph.edges.values())}\n"
    for from_v in graph.edges:
        for to_v in graph.edges[from_v]:
            w=graph.weights[(from_v,to_v)]
            result+=f"{from_v} -> {to_v} (权重：{w})\n"
    return result

def queryBridgeWords(graph,word1,word2):

    if word1 not in graph.vertices or word2 not in graph.vertices:
        if word1 not in graph.vertices and word2 not in graph.vertices:
            return f"No \"{word1}\" and \"{word2}\" in the graph!"
        elif word1 not in graph.vertices:
            return f"No \"{word1}\" in the graph!"
        else:
            return f"No \"{word2}\" in the graph!"
    bridge_words=[]

    for bridge in graph.vertices:
        if bridge in graph.edges.get(word1,[]) and word2 in graph.edges.get(bridge,[]):
            bridge_words.append(bridge)

    if not bridge_words:
        return f"No bridge words from \"{word1}\" to \"{word2}\"!"

    if len(bridge_words) == 1:
        return f"The bridge word from \"{word1}\" to \"{word2}\" is: {bridge_words[0]}"
    else:
        return f"The bridge words from \"{word1}\" to \"{word2}\" are: {', '.join(bridge_words[:-1])} and {bridge_words[-1]}"

def generateNewText(graph, inputText):
    words = re.findall(r'\b[a-z]+\b', inputText.lower())
    if len(words)<=1:
        return inputText
    result =[words[0]]
    for i in range(len(words)-1):
        word1 ,word2 = words[i], words[i+1]
        bridges=[b for b in graph.edges.get(word1,[]) if word2 in graph.edges.get(b,[])]
        if bridges:
            result.append(random.choice(bridges))
        result.append(word2)
    return ' '.join(result)
#可选功能，如果输入一个那么就展示所有，并且输出所有最短路径
def calcShortestPath(graph, word1, word2=None):
    if word1 not in graph.vertices:
        return f"No \"{word1}\" in the graph!"

    if word2 is None:
        # 计算 word1 到所有其他节点的最短路径，并显示路径长度
        results = []
        for word in graph.vertices:
            if word != word1:
                paths, length = dijkstra(graph, word1, word)
                for path in paths:
                    results.append(f"From \"{word1}\" to \"{word}\": {'->'.join(path)} (Length: {length})")
        return '\n'.join(results)

    # 如果给出第二个单词，计算 word1 到 word2 的最短路径
    paths, length = dijkstra(graph, word1, word2)
    return '\n'.join([f"{'->'.join(path)} (Length: {length})" for path in paths])


def reconstruct_paths(start, current, previous):
    """根据前驱节点重建所有最短路径"""
    if current == start:
        return [[start]]

    paths = []
    for predecessor in previous[current]:
        for path in reconstruct_paths(start, predecessor, previous):
            paths.append(path + [current])
    return paths

def dijkstra(graph, start, end):
    """Dijkstra算法：计算start到end的最短路径，并返回所有最短路径和路径的总长度"""
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start] = 0
    previous = {vertex: [] for vertex in graph.vertices}  # 允许多个前驱节点
    unvisited = set(graph.vertices)

    while unvisited:
        current = min(unvisited, key=lambda vertex: distances[vertex])

        if current == end or distances[current] == float('infinity'):
            break

        unvisited.remove(current)

        if current in graph.edges:
            for neighbor in graph.edges[current]:
                weight = graph.weights.get((current, neighbor), 1)
                distance = distances[current] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = [current]  # 新发现更短的路径，更新前驱节点
                elif distance == distances[neighbor]:
                    previous[neighbor].append(current)  # 如果发现相同长度的路径，保存前驱节点

    # 使用独立的 reconstruct_paths 函数重建所有最短路径
    paths = reconstruct_paths(start, end, previous)
    total_length = distances[end]  # 所有最短路径的长度都是相同的，等于计算出的最短距离

    return paths, total_length


def calPageRank(graph, word):
    """计算指定单词的PageRank值，阻尼系数d固定为0.85"""
    if word not in graph.vertices:
        return f"No \"{word}\" in the graph!"

    d = 0.85  # 固定阻尼系数
    max_iterations = 100
    tolerance = 1e-6
    vertices = graph.vertices
    n = len(vertices)
    if n == 0:
        return 0.0

    # 预处理：获取每个节点的出度和入边关系
    out_degree = {v: len(graph.edges[v]) for v in vertices}
    in_edges = defaultdict(list)
    for u in graph.edges:
        for v in graph.edges[u]:
            in_edges[v].append(u)
    dangling_nodes = [v for v in vertices if out_degree[v] == 0]

    # 初始化PageRank值
    pr = {v: 1.0 / n for v in vertices}

    for _ in range(max_iterations):
        new_pr = {}
        sum_pr_dangling = sum(pr[v] for v in dangling_nodes)

        for v in vertices:
            # 计算来自入边的贡献
            in_contrib = sum(pr[u] / out_degree[u] for u in in_edges[v] if out_degree[u] > 0)
            # 计算来自悬挂节点的贡献
            dangling_contrib = sum_pr_dangling / n if n > 0 else 0
            # 更新PageRank
            new_pr[v] = (1 - d) / n + d * (in_contrib + dangling_contrib)

        # 检查收敛
        delta = sum(abs(new_pr[v] - pr[v]) for v in vertices)
        if delta < tolerance:
            break
        pr = new_pr

    return pr.get(word, 0.0)


def randomWalk(graph):
    if not graph.vertices:
        return "Empty graph!"

    # 随机选择起始节点
    current = random.choice(graph.vertices)
    path = [current]
    visited_edges = set()  # 记录已访问的边

    print(f"随机游走从 {current} 开始...\n")

    while current in graph.vertices:
        # 查找当前节点的出边
        available_edges = [(current, next_v) for next_v in graph.edges[current]
                           if (current, next_v) not in visited_edges]

        # 如果没有可用的边，停止遍历
        if not available_edges:
            print("当前节点没有出边，游走结束。")
            break

        # 提示用户是否继续，按回车默认继续，否则退出
        user_input = input(f"当前节点：{current}，请选择是否继续游走（回车继续，输入 n 停止）：")
        if user_input.lower() == 'n':
            print("用户终止了游走。")
            break

        # 随机选择一条未访问的边
        from_v, to_v = random.choice(available_edges)
        visited_edges.add((from_v, to_v))  # 标记这条边为已访问
        path.append(to_v)
        current = to_v

    # 输出并保存结果到文件
    path_str = "->".join(path)

    # 将遍历结果写入文件
    with open("random_walk_result.txt", "w", encoding="utf-8") as file:
        file.write(path_str)

    return path_str


#可选功能，保存为图片
def saveGraphImage(graph, filepath='graph.png'):
    G = nx.DiGraph()
    for vertex in graph.vertices:
        G.add_node(vertex)
    for (from_v, to_v), weight in graph.weights.items():
        G.add_edge(from_v, to_v, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=2000, font_size=10, arrows=True, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Directed Graph")
    plt.savefig(filepath)
    plt.close()
    return f"图像已保存至 {filepath}"



def main():
    print("文本图结构处理系统")
    graph = Graph()
    filename = input("请输入文件路径: ")
    if not graph.build_from_text(filename):
        print("文件读取失败，请检查路径。")
        return

    while True:
        print("\n功能菜单:")
        print("1. 展示有向图")
        print("2. 查询桥接词")
        print("3. 生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank值")
        print("6. 随机游走")
        print("7. 保存图像到文件")
        print("0. 退出程序")
        choice = input("请选择功能 (0-7): ")

        if choice == '0':
            break
        elif choice == '1':
            print(showDirectedGraph(graph))
        elif choice == '2':
            word1 = input("请输入第一个单词: ").lower()
            word2 = input("请输入第二个单词: ").lower()
            print(queryBridgeWords(graph, word1, word2))
        elif choice == '3':
            inputText = input("请输入文本: ")
            print(generateNewText(graph, inputText))
        elif choice == '4':
            word1 = input("请输入起始单词: ").lower()
            word2= input("请输入目标单词 (留空则计算到所有单词): ").lower()
            if word2== "":
                print(calcShortestPath(graph, word1))  # 传入 None 或不传 word2
            else:
                print(calcShortestPath(graph, word1, word2))
        elif choice == '5':
            word = input("请输入单词: ").lower()
            pr = calPageRank(graph, word)
            if isinstance(pr, float):
                print(f"PageRank值: {pr:.6f}")
            else:
                print(pr)
        elif choice == '6':
            print(randomWalk(graph))
        elif choice == '7':
            path = input("请输入保存路径（默认 graph.png）: ").strip()
            if path == "":
                path = "graph.png"
            print(saveGraphImage(graph, path))
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    main()
