import numpy as np

class NaiveBayesClassifier:
    def __init__(self, smoothing=False):
        self.classes = None
        self.class_probs = {}
        self.feature_probs = {}
        self.smoothing = smoothing  
        self.logs = ""             

    def fit(self, X, y, feature_names=None):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        m = len(self.classes)  
        
        self.class_probs = {}
        self.feature_probs = {}
        
        for cls in self.classes:
            count_cls = np.sum(y == cls)
            if self.smoothing:
                self.class_probs[cls] = (count_cls + 1) / (n_samples + m)
            else:
                self.class_probs[cls] = count_cls / n_samples

        for cls in self.classes:
            X_cls = X[y == cls]
            n_samples_cls = len(X_cls)  
            self.feature_probs[cls] = []
            
            for feature_idx in range(n_features):
                unique_vals_all = np.unique(X[:, feature_idx])
                r = len(unique_vals_all) 
                
                col_data_cls = X_cls[:, feature_idx]
                value_probs = {}
                
                for val in unique_vals_all:
                    count_val = np.sum(col_data_cls == val)
                    
                    if self.smoothing:
                        prob = (count_val + 1) / (n_samples_cls + r)
                    else:
                        prob = count_val / n_samples_cls if n_samples_cls > 0 else 0.0
                        
                    value_probs[val] = prob
                    
                self.feature_probs[cls].append(value_probs)

    def predict_with_detailed_logging(self, X_predict, feature_names, target_name):
        predictions = []
        self.logs = ""  
        
        mode_str = "KỸ THUẬT LÀM TRƠN LAPLACE" if self.smoothing else "CHẾ ĐỘ THƯỜNG (NO SMOOTHING)"
        self.logs += f"THUẬT TOÁN NAÏVE BAYES PHÂN LỚP DỮ LIỆU - {mode_str}\n"
        self.logs += "=" * 80 + "\n\n"
        
        for idx, sample in enumerate(X_predict, 1):
            self.logs += f"👉 XÉT ĐỐI TƯỢNG CẦN DỰ ĐOÁN X{idx}:\n"
            sample_desc = ", ".join([f"{feature_names[i]} = '{sample[i]}'" for i in range(len(sample))])
            self.logs += f"   Mẫu X = {{ {sample_desc} }}\n"
            self.logs += f"   -----------------------------------------------------------------\n"
            
            class_scores = {}
            
            for cls in self.classes:
                self.logs += f"   * Xét giả thuyết nhãn [{target_name} = '{cls}']:\n"
                p_c = self.class_probs[cls]
                score = p_c
                
                formula_str = f"P({cls})"
                calc_str = f"{round(p_c, 4)}"
                
                for f_idx, f_val in enumerate(sample):
                    prob_dict = self.feature_probs[cls][f_idx]
                    prob = prob_dict.get(f_val, 17) 
                    
                    if not self.smoothing and prob == 0:
                        prob = 0.0
                        
                    score *= prob
                    formula_str += f" * P({feature_names[f_idx]}='{f_val}' | {cls})"
                    calc_str += f" * {round(prob, 4)}"
                
                class_scores[cls] = score
                self.logs += f"     ➔ Công thức: P({cls}|X) tỷ lệ với {formula_str}\n"
                self.logs += f"     ➔ Tính toán: {calc_str} = {round(score, 6)}\n\n"
            
            winner_class = max(class_scores, key=class_scores.get)
            predictions.append(winner_class)
            
            self.logs += f"   ➔ KẾT LUẬN: Đối tượng X{idx} thuộc lớp [{target_name} = '{winner_class}'] (Do có xác suất lớn hơn).\n"
            self.logs += "=" * 80 + "\n\n"
            
        return np.array(predictions), self.logs