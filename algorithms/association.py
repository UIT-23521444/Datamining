import pandas as pd
import numpy as np
from itertools import combinations

def calculate_support(transactions, itemsets):
    support_count = {}
    for itemset in itemsets:
        count = 0
        itemset_set = set(itemset) 
        for transaction in transactions:
            if itemset_set.issubset(set(transaction)):
                count += 1
        support_count[frozenset(itemset)] = count
    return support_count

def find_maximal_itemsets(frequent_itemsets):
    itemsets = [set(item["itemsets"]) for item in frequent_itemsets]
    maximal_itemsets = []
    for i, itemset in enumerate(itemsets):
        is_maximal = all(not itemset < other for j, other in enumerate(itemsets) if i != j)
        if is_maximal:
            maximal_itemsets.append(frequent_itemsets[i])
    return maximal_itemsets

def apriori(transactions, min_support):
    cleaned_trans = []
    for t in transactions:
        clean_row = [str(item).strip() for item in t if str(item).strip() not in ['', 'nan', 'None', 'NaN']]
        if clean_row:
            cleaned_trans.append(clean_row)
    
    num_trans = len(cleaned_trans)
    single_items = list(set(item for t in cleaned_trans for item in t))
    current_itemsets = [[item] for item in single_items]
    frequent_itemsets = []

    while current_itemsets:
        support_counts = calculate_support(cleaned_trans, current_itemsets)
        
        filtered = []
        for itemset_fs, count in support_counts.items():
            sup = count / num_trans
            if sup >= min_support:
                filtered.append({"itemsets": itemset_fs, "support": sup})
        
        if not filtered:
            break
            
        frequent_itemsets.extend(filtered)
        
        current_lk = [sorted(list(f["itemsets"])) for f in filtered]
        next_candidates = []
        lk_size = len(current_lk)
        k = len(current_itemsets[0])
        
        for i in range(lk_size):
            for j in range(i + 1, lk_size):
                if current_lk[i][:k-1] == current_lk[j][:k-1]:
                    candidate = sorted(list(set(current_lk[i] + current_lk[j])))
                    if candidate not in next_candidates:
                        next_candidates.append(candidate)
                        
        current_itemsets = next_candidates
    
    maximal = find_maximal_itemsets(frequent_itemsets)
    return frequent_itemsets, maximal

def generate_rules(frequent_itemsets, min_confidence):
    """Sinh luật kết hợp toàn diện (Cố định thứ tự và đồng bộ Key)"""
    support_table = {item["itemsets"]: item["support"] for item in frequent_itemsets}
    rules = []
    
    for item in frequent_itemsets:
        itemset = item["itemsets"] 
        if len(itemset) < 2:
            continue
            
        sorted_item_list = sorted(list(itemset))
        
        for r in range(1, len(sorted_item_list)):
            for ante in combinations(sorted_item_list, r):
                ante_fs = frozenset(ante)
                cons_fs = itemset - ante_fs 
                
                if ante_fs in support_table:
                    conf = support_table[itemset] / support_table[ante_fs]
                    
                    if conf >= min_confidence:
                        rules.append({
                            "antecedents": ante_fs,
                            "consequents": cons_fs,
                            "support": support_table[itemset],
                            "confidence": conf
                        })
    return rules