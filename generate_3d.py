import networkx as nx
import plotly.graph_objects as go
import os

from Lab_Day19_GraphRAG import load_data, DATASET_DIR, MAX_FILES, build_knowledge_graph

def draw_3d_graph(G, output_path="knowledge_graph_3d.html"):
    print(">>> Đang tính toán toạ độ 3D cho đồ thị...")
    # Sử dụng spring_layout với 3 chiều (dim=3)
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    z_nodes = [pos[k][2] for k in G.nodes()]
    
    x_edges = []
    y_edges = []
    z_edges = []
    
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])
        
    trace_edges = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    )
    
    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=6, color='skyblue', line=dict(width=2, color='DarkSlateGrey')),
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[trace_edges, trace_nodes])
    fig.update_layout(
        title="Knowledge Graph 3D Visualization",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_path)
    print(f">>> Đã xuất đồ thị 3D ra file: {output_path}")

if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("Lỗi: Không tìm thấy GROQ_API_KEY")
        
    chunks = load_data(DATASET_DIR, MAX_FILES)
    G, _ = build_knowledge_graph(chunks)
    draw_3d_graph(G, "knowledge_graph_3d.html")
