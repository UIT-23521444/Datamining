def calculate_pearson(x, y):
    if len(x) != len(y):
        return 0, "Độ dài hai chuỗi không khớp."
    
    n = len(x)
    
    try:
        x = [float(i) for i in x]
        y = [float(i) for i in y]
    except ValueError:
        return 0, "Dữ liệu chứa ký tự chữ"
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = (sum((x[i] - mean_x)**2 for i in range(n)) * sum((y[i] - mean_y)**2 for i in range(n)))**0.5
    
    if den == 0:
        return 0, "Biến không đổi, không thể tính tương quan."
    
    r = num / den
    r_round = round(r, 4)
    if r_round == 1:
        text = "Tương quan thuận tuyệt đối."
    elif 0.7 <= r_round < 1:
        text = "Tương quan thuận rất mạnh."
    elif 0.3 <= r_round < 0.7:
        text = "Tương quan thuận ở mức trung bình."
    elif 0 < r_round < 0.3:
        text = "Tương quan thuận yếu."
    elif r_round == 0:
        text = "Không có tương quan tuyến tính."
    elif -0.3 < r_round < 0:
        text = "Tương quan nghịch yếu."
    elif -0.7 <= r_round <= -0.3:
        text = "Tương quan nghịch ở mức trung bình."
    elif -1 < r_round <= -0.7:
        text = "Tương quan nghịch rất mạnh."
    else:
        text = "Tương quan nghịch tuyệt đối."

    return r_round, text