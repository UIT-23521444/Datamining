import pandas as pd
import numpy as np

def generate_all_combinations(attributes):
    combinations = []
    def combine(current, start):
        if current:
            combinations.append(current)
        for i in range(start, len(attributes)):
            combine(current + [attributes[i]], i + 1)
    combine([], 0)
    return combinations

def indiscernibility_relation(data, attributes):
    id_col = data.columns[0] 
    groups = data.groupby(attributes).groups
    
    return [[str(x) for x in data.loc[list(indices), id_col].tolist()] for indices in groups.values()]

def lower_approximation(target_set, ind_relation):
    lower = []
    for subset in ind_relation:
        if set(subset).issubset(target_set):
            lower.extend(subset)
    return lower

def upper_approximation(target_set, ind_relation):
    upper = []
    for subset in ind_relation:
        if set(subset) & set(target_set):
            upper.extend(subset)
    return upper

def calculate_rough_accuracy(lower, upper):
    return len(lower) / len(upper) if len(upper) > 0 else 0

def discernibility_matrix(data):
    n = len(data)
    attributes = list(data.columns[1:-1]) 
    matrix = [[set() for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if data.iloc[i, -1] != data.iloc[j, -1]:
                diff_attributes = {
                    attr for attr in attributes
                    if data.iloc[i][attr] != data.iloc[j][attr]
                }
                matrix[i][j] = diff_attributes
    return matrix

def get_reducts(attributes, matrix):
    conditions = set()
    for row in matrix:
        for cell in row:
            if cell:
                conditions.add(frozenset(cell))
    
    cond_list = list(conditions)
    reducts = []

    def satisfies(subset):
        subset_set = set(subset)
        return all(any(attr in subset_set for attr in cond) for cond in cond_list)

    for size in range(1, len(attributes) + 1):
        for subset in generate_all_combinations(attributes):
            if len(subset) == size and satisfies(subset):
                
                is_minimal = True
                for r in reducts:
                    if set(r).issubset(set(subset)):
                        is_minimal = False
                        break
                
                if is_minimal:
                    reducts.append(list(subset))
                    
    return reducts

def generate_rough_rules(data, reduct):
    rules = []
    decision_col = data.columns[-1]
    for _, group in data.groupby(reduct):
        decisions = group[decision_col].unique()
        if len(decisions) == 1:
            condition = " AND ".join([f"{col}={group.iloc[0][col]}" for col in reduct])
            rules.append(f"IF {condition} THEN {decision_col}={decisions[0]}")
    return rules

def run_rough_set_analysis(dataset):
    id_col = dataset.columns[0]
    attributes = list(dataset.columns[1:-1])
    decision_col = dataset.columns[-1]
    decision_classes = dataset[decision_col].unique()
    
    ind_relation = indiscernibility_relation(dataset, attributes)
    approx_results = []
    
    total_lower_size = 0
    total_objects = len(dataset)
    
    for cls in decision_classes:
        target_set = [str(x) for x in dataset[dataset[decision_col] == cls][id_col].tolist()]
        
        lower = lower_approximation(target_set, ind_relation)
        upper = upper_approximation(target_set, ind_relation)
        acc = calculate_rough_accuracy(lower, upper)
        
        total_lower_size += len(lower)
        
        approx_results.append({
            'Lớp': cls,
            'Xấp xỉ dưới': str(lower),
            'Xấp xỉ trên': str(upper),
            'Độ chính xác': f"{acc*100:.2f}%"
        })

    dependency_k = total_lower_size / total_objects if total_objects > 0 else 0
    approx_results.append({
        'Lớp': 'TOÀN BỘ (Hệ số phụ thuộc k)',
        'Xấp xỉ dưới': '-',
        'Xấp xỉ trên': '-',
        'Độ chính xác': f"{dependency_k*100:.2f}%"
    })

    matrix = discernibility_matrix(dataset)
    reducts = get_reducts(attributes, matrix)

    all_rules = []
    for red in reducts:
        all_rules.extend(generate_rough_rules(dataset, red))

    all_rules = list(set(all_rules))
        
    return approx_results, reducts, all_rules