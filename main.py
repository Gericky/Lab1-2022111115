import re
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = defaultdict(list)
        self.weights = {}
        self.word_counts = defaultdict(int)  # 添加词频统计

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)

    def add_edge(self, from_vertex, to_vertex):
        if from_vertex not in self.vertices:
            self.add_vertex(from_vertex)
        if to_vertex not in self.vertices:
            self.add_vertex(to_vertex)
        # 检查 to_vertex 是否已在邻接列表中，如果不在则添加
        if to_vertex not in self.edges[from_vertex]:
            self.edges[from_vertex].append(to_vertex)
        # 总是更新或设置边的权重
        edge = (from_vertex, to_vertex)
        self.weights[edge] = self.weights.get(edge, 0) + 1

    def build_from_text(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                words = re.findall(r'\b[a-z]+\b', text)
                if not words:
                    return False  # 如果没有单词，则构建失败
                # 统计词频
                for word in words:
                    self.word_counts[word] += 1
                # 构建图的边
                for i in range(len(words) - 1):
                    self.add_edge(words[i], words[i + 1])
                # 确保所有计数的单词都作为顶点存在（即使它们没有边）
                for word in self.word_counts:
                    self.add_vertex(word)
                return True
        except Exception:
            return False


def showDirectedGraph(graph):
    result = "有向图结构如下：\n"
    result += f"顶点数：{len(graph.vertices)}\n"
    result += f"边数：{sum(len(v) for v in graph.edges.values())}\n"
    for from_v in graph.edges:
        for to_v in graph.edges[from_v]:
            w = graph.weights[(from_v, to_v)]
            result += f"{from_v} -> {to_v} (权重：{w})\n"
    return result


def queryBridgeWords(graph, word1, word2):
    if word1 not in graph.vertices or word2 not in graph.vertices:
        if word1 not in graph.vertices and word2 not in graph.vertices:
            return f"No \"{word1}\" and \"{word2}\" in the graph!"
        elif word1 not in graph.vertices:
            return f"No \"{word1}\" in the graph!"
        else:
            return f"No \"{word2}\" in the graph!"
    bridge_words = []

    for bridge in graph.vertices:
        if bridge in graph.edges.get(word1, []) and word2 in graph.edges.get(bridge, []):
            bridge_words.append(bridge)

    if not bridge_words:
        return f"No bridge words from \"{word1}\" to \"{word2}\"!"

    if len(bridge_words) == 1:
        return f"The bridge word from \"{word1}\" to \"{word2}\" is: {bridge_words[0]}"
    else:
        return f"The bridge words from \"{word1}\" to \"{word2}\" are: {', '.join(bridge_words[:-1])} and {bridge_words[-1]}"


def generateNewText(graph, inputText):
    words = re.findall(r'\b[a-z]+\b', inputText.lower())
    if len(words) <= 1:
        return inputText
    result = [words[0]]
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        bridges = [b for b in graph.edges.get(word1, []) if word2 in graph.edges.get(b, [])]
        if bridges:
            result.append(random.choice(bridges))
        result.append(word2)
    return ' '.join(result)


def calcShortestPath(graph, word1, word2=None):
    if word1 not in graph.vertices:
        return f"No \"{word1}\" in the graph!"

    if word2 is None:
        results = []
        for word in graph.vertices:
            if word != word1:
                paths, length = dijkstra(graph, word1, word)
                for path in paths:
                    results.append(f"From \"{word1}\" to \"{word}\": {'->'.join(path)} (Length: {length})")
        return '\n'.join(results)

    paths, length = dijkstra(graph, word1, word2)
    return '\n'.join([f"{'->'.join(path)} (Length: {length})" for path in paths])


def reconstruct_paths(start, current, previous):
    if current == start:
        return [[start]]

    paths = []
    for predecessor in previous[current]:
        for path in reconstruct_paths(start, predecessor, previous):
            paths.append(path + [current])
    return paths


def dijkstra(graph, start, end):
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start] = 0
    previous = {vertex: [] for vertex in graph.vertices}
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
                    previous[neighbor] = [current]
                elif distance == distances[neighbor]:
                    previous[neighbor].append(current)

    paths = reconstruct_paths(start, end, previous)
    total_length = distances[end]

    return paths, total_length


def calPageRank(graph, word):
    d = 0.85
    max_iter = 100
    tol = 1e-6
    use_word_frequency_init = True
    if word not in graph.vertices:
        return f"No \"{word}\" in the graph!"

    vertices = graph.vertices
    num_vertices = len(vertices)
    if num_vertices == 0:
        return 0.0

    if use_word_frequency_init and graph.word_counts:
        total_word_count = sum(graph.word_counts.values())
        if total_word_count > 0:
            pr = {vertex: graph.word_counts.get(vertex, 0) / total_word_count for vertex in vertices}
            current_sum = sum(pr.values())
            if current_sum > 0:
                epsilon = 1e-9
                pr = {v: (p / current_sum) + epsilon for v, p in pr.items()}
                final_sum = sum(pr.values())
                pr = {v: p / final_sum for v, p in pr.items()}
            else:
                pr = {vertex: 1.0 / num_vertices for vertex in vertices}
        else:
            pr = {vertex: 1.0 / num_vertices for vertex in vertices}
    else:
        pr = {vertex: 1.0 / num_vertices for vertex in vertices}

    out_degree = {vertex: len(graph.edges.get(vertex, [])) for vertex in vertices}

    for iteration in range(max_iter):
        prev_pr = pr.copy()
        new_pr = {vertex: 0.0 for vertex in vertices}

        dangling_sum = 0
        for vertex in vertices:
            if out_degree[vertex] == 0:
                dangling_sum += prev_pr[vertex]

        dangling_contribution_per_node = d * dangling_sum / num_vertices

        for vertex in vertices:
            teleport_prob = (1 - d) / num_vertices

            inlink_contribution = 0
            for linker in vertices:
                if vertex in graph.edges.get(linker, []):
                    linker_out_degree = out_degree.get(linker, 0)
                    if linker_out_degree > 0:
                        inlink_contribution += d * (prev_pr[linker] / linker_out_degree)

            new_pr[vertex] = teleport_prob + dangling_contribution_per_node + inlink_contribution

        pr = new_pr

        diff = sum(abs(pr[v] - prev_pr[v]) for v in vertices)
        if diff < tol:
            break

    return pr.get(word, 0.0)


def randomWalk(graph):
    if not graph.vertices:
        return "Empty graph!"

    current = random.choice(graph.vertices)
    path = [current]
    visited_edges = set()

    print(f"随机游走从 {current} 开始...\n")

    while current in graph.vertices:
        available_edges = [(current, next_v) for next_v in graph.edges[current]
                           if (current, next_v) not in visited_edges]

        if not available_edges:
            print("当前节点没有出边，游走结束。")
            break

        user_input = input(f"当前节点：{current}，按回车继续，输入任意字符后回车停止：")
        if user_input != '':
            print("用户终止了游走。")
            break

        from_v, to_v = random.choice(available_edges)
        visited_edges.add((from_v, to_v))
        path.append(to_v)
        current = to_v

    path_str = "->".join(path)

    with open("random_walk_result.txt", "w", encoding="utf-8") as file:
        file.write(path_str)

    return path_str


def saveGraphImage(graph, filepath='graph.png'):
    G = nx.DiGraph()
    for vertex in graph.vertices:
        G.add_node(vertex)
    for (from_v, to_v), weight in graph.weights.items():
        G.add_edge(from_v, to_v, weight=weight)

    pos = nx.kamada_kawai_layout(G)  # 尝试 Kamada-Kawai layout，通常能更好地分散节点
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(20, 15))  # 保持较大的画布尺寸
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=1500,  # 显著减小节点大小
            font_size=9,  # 节点标签字体大小
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20)  # 调整箭头大小
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3,
                                 font_size=8,  # 边标签字体大小
                                 font_color='darkred',  # 使用深色以便区分
                                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))  # 添加背景框减少遮挡

    plt.title("Directed Graph Visualization")
    plt.margins(0.1)  # 可以尝试添加一些边距
    plt.savefig(filepath, dpi=300)
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
            word2 = input("请输入目标单词 (留空则计算到所有单词): ").lower()
            if word2 == "":
                print(calcShortestPath(graph, word1))
            else:
                print(calcShortestPath(graph, word1, word2))
        elif choice == '5':
            word = input("请输入单词: ").lower()
            pr = calPageRank(graph, word)
            if isinstance(pr, float):
                print(f"PageRank值 (基于词频初始化): {pr:.6f}")
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
