def calculate_pearson(x, y):
    if len(x) != len(y):
        return 0, "Độ dài hai chuỗi không khớp."
    
    n = len(x)
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = (sum((x[i] - mean_x)**2 for i in range(n)) * sum((y[i] - mean_y)**2 for i in range(n)))**0.5
    
    if den == 0:
        return 0, "Biến không đổi, không thể tính tương quan."
    
    r = num / den
    
    if abs(r) >= 0.7: text = "Tương quan rất chặt chẽ."
    elif abs(r) >= 0.3: text = "Tương quan mức độ trung bình."
    else: text = "Tương quan yếu hoặc không có."
    
    return round(r, 4), text