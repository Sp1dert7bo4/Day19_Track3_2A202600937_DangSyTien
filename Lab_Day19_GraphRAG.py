import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import glob
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Removed RetrievalQA import
from langchain_core.prompts import PromptTemplate

# ==========================================
# CẤU HÌNH API KEY VÀ THAM SỐ
# ==========================================
# API Key được nạp từ biến môi trường của hệ thống
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix huggingface warning

DATASET_DIR = "dataset/dataset/"
# Chạy toàn bộ corpus
MAX_FILES = 100 
LLM_MODEL = "llama-3.3-70b-versatile"

def get_llm():
    return ChatGroq(temperature=0, model_name=LLM_MODEL)

# ==========================================
# PHẦN 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_data(data_dir: str, max_files: int) -> List[Document]:
    print(">>> [1] Đang đọc dữ liệu từ corpus...")
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    documents = []
    
    for idx, file_path in enumerate(files[:max_files]):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Tách lấy phần Full Content
            if "Full Content:" in content:
                full_content = content.split("Full Content:")[1].strip()
            else:
                full_content = content.strip()
                
            if full_content:
                documents.append(Document(page_content=full_content, metadata={"source": file_path}))
                
    print(f"Đã đọc {len(documents)} tài liệu.")
    
    # Chia nhỏ văn bản (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn (chunks).")
    return chunks

# ==========================================
# PHẦN 2: XÂY DỰNG FLAT RAG (BASELINE)
# ==========================================
def build_flat_rag(chunks: List[Document]):
    print(">>> [2] Đang xây dựng hệ thống Flat RAG (Vector Database)...")
    # Thay OpenAI Embedding bằng mô hình Open Source (miễn phí, chạy local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def flat_rag_query(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Dựa vào các thông tin sau đây:\n{context}\n\nHãy trả lời câu hỏi: {query}"
    )
    llm = get_llm()
    answer = (prompt | llm).invoke({"context": context, "query": query}).content
    return answer

# ==========================================
# PHẦN 3: XÂY DỰNG GRAPHRAG
# ==========================================
def extract_triples(llm, text: str) -> List[Tuple[str, str, str]]:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Bạn là một chuyên gia về trích xuất thông tin (Information Extraction).
Nhiệm vụ của bạn là đọc đoạn văn bản sau và trích xuất các thông tin dưới dạng các bộ ba (Subject, Predicate, Object).
Chỉ xuất ra các bộ ba, mỗi bộ ba trên một dòng theo định dạng: (Subject, Predicate, Object). Không thêm bất kỳ văn bản giải thích nào khác.
Ví dụ:
(OpenAI, FOUNDED_BY, Sam Altman)
(Tesla, PRODUCES, Electric Vehicles)

Văn bản:
{text}

Triples:"""
    )
    chain = prompt | llm
    response = chain.invoke({"text": text}).content
    
    triples = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            line = line[1:-1]
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                subj = parts[0]
                pred = parts[1]
                obj = ", ".join(parts[2:]) 
                triples.append((subj, pred, obj))
    return triples

def build_knowledge_graph(chunks: List[Document]) -> Tuple[nx.DiGraph, dict]:
    print(">>> [3] Đang xây dựng hệ thống GraphRAG...")
    llm = get_llm()
    G = nx.DiGraph()
    
    start_time = time.time()
    
    print(f"Đang trích xuất thực thể cho toàn bộ {len(chunks)} chunks bằng Groq... (Quá trình này có thể mất thời gian)")
    
    for idx, chunk in enumerate(chunks):
        if (idx + 1) % 10 == 0:
            print(f"  -> Đã trích xuất {idx + 1}/{len(chunks)} chunks...")
        try:
            triples = extract_triples(llm, chunk.page_content)
            for subj, pred, obj in triples:
                G.add_edge(subj, obj, label=pred)
        except Exception as e:
            print(f"Lỗi ở chunk {idx}: {e}")
            time.sleep(5)
        time.sleep(0.5)

    end_time = time.time()
    
    cost_info = {
        "tokens": "Không tính phí với Groq free tier",
        "cost_usd": 0.0,
        "time_seconds": end_time - start_time
    }
    print(f"Đã xây dựng đồ thị với {G.number_of_nodes()} nodes và {G.number_of_edges()} edges.")
    print(f"Thời gian xây dựng Graph bằng Groq: {cost_info['time_seconds']:.2f}s")
    
    return G, cost_info

def draw_graph(G: nx.DiGraph, output_path="knowledge_graph.png"):
    print(">>> Đang vẽ và lưu đồ thị...")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Knowledge Graph from Tech Company Corpus")
    plt.savefig(output_path)
    print(f"Đã lưu ảnh đồ thị tại: {output_path}")

def graph_rag_query(G: nx.DiGraph, query: str) -> str:
    llm = get_llm()
    
    # 1. Trích xuất Entity chính từ Query
    prompt_extract = PromptTemplate(
        input_variables=["query"],
        template="Xác định (các) thực thể chính (Entity) trong câu hỏi sau. Chỉ trả về tên thực thể, cách nhau bởi dấu phẩy. Câu hỏi: {query}"
    )
    entities_str = (prompt_extract | llm).invoke({"query": query}).content
    entities = [e.strip() for e in entities_str.split(',')]
    
    # 2. Graph Traversal (2-hops)
    context_triples = []
    for entity in entities:
        matched_nodes = [n for n in G.nodes() if entity.lower() in str(n).lower()]
        for start_node in matched_nodes:
            for neighbor in G.neighbors(start_node):
                edge_data = G.get_edge_data(start_node, neighbor)
                context_triples.append(f"({start_node}, {edge_data.get('label', '')}, {neighbor})")
                for second_neighbor in G.neighbors(neighbor):
                    edge_data_2 = G.get_edge_data(neighbor, second_neighbor)
                    context_triples.append(f"({neighbor}, {edge_data_2.get('label', '')}, {second_neighbor})")
    
    context_triples = list(set(context_triples))
    context_text = "\n".join(context_triples)
    
    if not context_text:
        context_text = "Không tìm thấy thông tin trong đồ thị tri thức."
        
    prompt_qa = PromptTemplate(
        input_variables=["context", "query"],
        template="Dựa vào các thông tin tri thức (Triples) sau đây:\n{context}\n\nHãy trả lời câu hỏi một cách ngắn gọn: {query}\nNếu không đủ thông tin, hãy nói 'Không có thông tin'."
    )
    answer = (prompt_qa | llm).invoke({"context": context_text, "query": query}).content
    return answer

# ==========================================
# PHẦN 4: ĐÁNH GIÁ (EVALUATION)
# ==========================================
def evaluate_systems(flat_qa_chain, G: nx.DiGraph):
    print(">>> [4] Đang thực hiện Evaluation với 20 câu hỏi benchmark...")
    
    questions = [
        "What is the main topic of the ICCT study published in September 2021?",
        "Who authored the briefing on September 14, 2021?",
        "How many electric vehicles were sold annually in the US from 2018 to 2020?",
        "What was the electric share of new vehicle sales in 2020?",
        "Which regulations are essential to electric vehicle market growth according to the study?",
        "What was the combined new electric vehicle share in states with ZEV regulations?",
        "Which metric slowed down in Q1 2024 according to Cox Automotive?",
        "What was the EV share of total new-vehicle sales in Q1 2024?",
        "Who is the director of Industry Insights at Cox Automotive?",
        "What did Stephanie Valdez Streaty say about electric vehicle sales in Q1 2024?",
        "How much did Tesla sales in the U.S. decrease year over year in Q1 2024?",
        "What was Tesla's share of the electric vehicle market in Q1 2024?",
        "Which nine manufacturers recorded more than 50% year-over-year growth in EV sales?",
        "What was the average transaction price for a new EV in Q1 2024?",
        "What percentage of all EVs were leased in Q1 2024?",
        "Which Cadillac model drove a 499.2% year-over-year increase in EV sales?",
        "What happened to the sales of the Chevy Bolt in Q1 2024?",
        "When is the new version of the Chevy Bolt expected to launch?",
        "Which automaker achieved an 86.1% year-over-year increase in Q1 EV sales?",
        "What is Cox Automotive's forecast for EV sales in the U.S. by the end of 2024?"
    ]
    
    results = []
    # Chỉ chạy 5 câu để test tốc độ nếu không muốn đợi quá lâu, nhưng đề yêu cầu 20 câu
    for i, q in enumerate(questions):
        print(f"Đang xử lý câu hỏi {i+1}/20...")
        ans_flat = flat_rag_query(flat_qa_chain, q)
        ans_graph = graph_rag_query(G, q)
        
        results.append({
            "Question": q,
            "Flat_RAG_Answer": ans_flat,
            "GraphRAG_Answer": ans_graph
        })
        time.sleep(1) # Tránh rate limit của Groq
        
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
    print(">>> Đã lưu kết quả so sánh tại benchmark_results.csv")
    return df

if __name__ == "__main__":
    chunks = load_data(DATASET_DIR, MAX_FILES)
    qa_chain = build_flat_rag(chunks)
    
    G, cost_info = build_knowledge_graph(chunks)
    draw_graph(G, "knowledge_graph.png")
    
    df_results = evaluate_systems(qa_chain, G)
    
    print("\n--- HOÀN THÀNH LAB DAY 19 ---")
    print(f"Xem file 'knowledge_graph.png' để thấy đồ thị tri thức.")
    print(f"Xem file 'benchmark_results.csv' để xem kết quả so sánh 20 câu hỏi.")
