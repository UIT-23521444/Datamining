import numpy as np
import pandas as pd
import pydot
import uuid
import subprocess
import os

def find_entropy(df):
    target = df.keys()[-1] 
    total_samples = len(df)
    
    if total_samples == 0:
        return 0
        
    _, counts = np.unique(df[target], return_counts=True)
    entropy = 0
    
    for count in counts:
        fraction = count / total_samples
        if fraction > 0:
            entropy += -fraction * np.log2(fraction)
            
    return entropy


def find_entropy_attribute(df, attribute):
    target = df.keys()[-1]
    variables = df[attribute].unique()
    entropy_attribute = 0
    total_samples = len(df)
    
    for variable in variables:
        subtable = df[df[attribute] == variable]
        den = len(subtable)
        
        if den == 0:
            continue
            
        entropy_v = find_entropy(subtable)
        weight = den / total_samples
        entropy_attribute += weight * entropy_v
        
    return abs(entropy_attribute)


def find_winner(df, current_node_name, logs_list):
    attributes = df.keys()[:-1]
    base_entropy = find_entropy(df)
    step_log = f"--- TÍNH TOÁN TẠI PHÂN NHÁNH ID3: [{current_node_name}] (Tổng mẫu: {len(df)}) ---\n"
    step_log += f" * Entropy gốc (Độ bất định) của tập dữ liệu tại nút này I(p,n) = {round(base_entropy, 5)}\n"
    step_log += " * Khảo sát các thuộc tính ứng viên:\n"
    
    gains = []
    
    for key in attributes:
        e_attr = find_entropy_attribute(df, key)
        gain = base_entropy - e_attr
        gain_round = round(gain, 5)
        gains.append(gain_round)
        
        step_log += f"   ➔ Thuộc tính [{key}]: E({key}) = {round(e_attr, 5)} | Gain({key}) = {gain_round}\n"
        
    winner_idx = np.argmax(gains)
    winner_attr = attributes[winner_idx]
    
    step_log += f" ➔ THUỘC TÍNH CHIẾN THẮNG ĐƯỢC CHỌN: [{winner_attr}] (Do có Độ lợi thông tin Gain lớn nhất: {gains[winner_idx]})\n"
    step_log += "-" * 75 + "\n\n"
    
    logs_list.append(step_log)
    return winner_attr


def build_tree_with_logs(df, current_node_name="Gốc", logs_list=None):
    if logs_list is None:
        logs_list = []
        
    target = df.keys()[-1]

    if len(df.columns) == 1:
        return df[target].mode()[0], logs_list

    clValue, counts = np.unique(df[target], return_counts=True)
    if len(counts) == 1:
        return clValue[0], logs_list
        
    node = find_winner(df, current_node_name, logs_list)
    att_values = np.unique(df[node])
    
    tree = {node: {}}
    
    for value in att_values:
        subtable = df[df[node] == value].reset_index(drop=True)
        subtable = subtable.drop(columns=[node])
        
        if len(subtable) == 0:
            tree[node][value] = df[target].mode()[0]
            continue
            
        sub_clValue, sub_counts = np.unique(subtable[target], return_counts=True)
        
        if len(sub_counts) == 1: 
            tree[node][value] = sub_clValue[0]
            logs_list.append(f" ➔ ĐỊNH HƯỚNG PHÂN NHÁNH THUẦN NHẤT: {node} = {value} -> Kết luận nhãn: {sub_clValue[0]}\n\n")
        else:
            next_node_name = f"{current_node_name} ➔ {node}={value}"
            subtree, _ = build_tree_with_logs(subtable, next_node_name, logs_list)
            tree[node][value] = subtree
            
    return tree, logs_list

def save_tree_image(tree_dict, filename):
    graph = pydot.Dot(graph_type='digraph', charset='utf-8')
    graph.set_node_defaults(fontname="Arial", fontsize="11")
    graph.set_edge_defaults(fontname="Arial", fontsize="10")

    def walk(tree, parent_uid=None, edge_label=""):
        if isinstance(tree, dict):
            for node_name, branches in tree.items():
                current_uid = str(uuid.uuid4())
                node = pydot.Node(current_uid, label=str(node_name), shape='oval', style='filled', fillcolor='#e3f2fd', color='#2196f3')
                graph.add_node(node)
                
                if parent_uid:
                    graph.add_edge(pydot.Edge(parent_uid, current_uid, label=str(edge_label)))
                
                if isinstance(branches, dict):
                    for branch_value, subtree in branches.items():
                        walk(subtree, current_uid, branch_value)
        else:
            leaf_uid = str(uuid.uuid4())
            leaf = pydot.Node(leaf_uid, label=str(tree), shape='box', style='filled', fillcolor='#e8f5e9', color='#4caf50', fontweight='bold')
            graph.add_node(leaf)
            
            if parent_uid:
                graph.add_edge(pydot.Edge(parent_uid, leaf_uid, label=str(edge_label)))

    if tree_dict:
        walk(tree_dict)

    temp_dot = "temp_tree_id3.dot"
    try:
        with open(temp_dot, "w", encoding="utf-8") as f:
            f.write(graph.to_string())
        subprocess.run(["dot", "-Tpng", temp_dot, "-o", filename], check=True)
        if os.path.exists(temp_dot):
            os.remove(temp_dot)
        return True
    except Exception as e:
        if os.path.exists(temp_dot):
            os.remove(temp_dot)
        return False

build_tree = build_tree_with_logs
def extract_rules(tree, current_rule=None, rules_list=None):
    if rules_list is None:
        rules_list = []
    if current_rule is None:
        current_rule = []

    if isinstance(tree, dict):
        for node, branches in tree.items():
            for branch_value, subtree in branches.items():
                condition = f"[{node}] = '{branch_value}'"
                extract_rules(subtree, current_rule + [condition], rules_list)
    else:
        if current_rule:
            if_part = " AND ".join(current_rule)
            then_part = f"➔ [{tree}]"
            rules_list.append(f" ➔ IF  {if_part}  THEN  {then_part}")
        else:
            rules_list.append(f" ➔ Kết luận trực tiếp: {tree}")

    return rules_list