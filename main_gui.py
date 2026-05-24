import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from PIL import Image, ImageTk
import platform
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
from algorithms import build_tree_with_logs, save_tree_image
from algorithms.decision_tree import extract_rules

graphviz_paths = [
    r'C:\Program Files\Graphviz\bin',
    r'C:\Program Files (x86)\Graphviz\bin',
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Graphviz", "bin")
]

for p in graphviz_paths:
    if os.path.exists(p):
        os.environ["PATH"] += os.pathsep + p
        break

from algorithms import (
    build_tree_with_logs, 
    save_tree_image,
    KMeansAlgorithm,
    calculate_pearson,
    NaiveBayesClassifier,
    run_rough_set_analysis,
    apriori, 
    generate_rules
)

class DataMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UIT - Công Cụ Khai Thác Dữ Liệu Đa Thuật Toán")
        self.root.geometry("1150x900")
        self.root.configure(bg="#f4f7f6")

        self.df = None
        self.column_names = []
        
        self.photo = None 

        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.main_container, 
            bg="#f4f7f6", 
            highlightthickness=0
        )
        
        self.scrollbar = tk.Scrollbar(
            self.main_container, 
            orient="vertical", 
            command=self.canvas.yview
        )
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f4f7f6")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.setup_ui()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def setup_ui(self):
        tk.Label(
            self.scrollable_frame, 
            text="PHẦN MỀM KHAI THÁC DỮ LIỆU ĐA THUẬT TOÁN", 
            fg="#2c3e50", 
            bg="#f4f7f6", 
            font=("Arial", 18, "bold")
        ).pack(pady=15, fill="x")

        form_frame = tk.Frame(
            self.scrollable_frame, 
            bg="white", 
            bd=1, 
            relief="solid", 
            padx=20, 
            pady=20
        )
        form_frame.pack(padx=40, fill="x")

        top_btn_frame = tk.Frame(form_frame, bg="white")
        top_btn_frame.pack(fill="x")
        
        tk.Button(
            top_btn_frame, 
            text="📁 Chọn Tệp CSV", 
            command=self.upload_file, 
            bg="#34495e", 
            fg="white", 
            font=("Arial", 10, "bold"), 
            relief="flat", 
            padx=20, 
            pady=8
        ).pack(side="left")
        
        self.lbl_file = tk.Label(
            top_btn_frame, 
            text="Chưa có dữ liệu được chọn", 
            bg="white", 
            fg="#e74c3c", 
            font=("Arial", 10, "italic")
        )
        self.lbl_file.pack(side="left", padx=15)

        param_frame = tk.LabelFrame(
            form_frame, 
            text=" Cài đặt Tham số & Chọn Cột ", 
            bg="white", 
            font=("Arial", 10, "bold"), 
            padx=10, 
            pady=10
        )
        param_frame.pack(fill="x", pady=15)

        tk.Label(param_frame, text="Cột X (Số):", bg="white").grid(row=0, column=0, sticky="e", pady=5)
        self.cb_col_x = ttk.Combobox(param_frame, width=25, state="readonly")
        self.cb_col_x.grid(row=0, column=1, padx=10, sticky="w")

        tk.Label(param_frame, text="Cột Y (Số):", bg="white").grid(row=0, column=2, sticky="e")
        self.cb_col_y = ttk.Combobox(param_frame, width=25, state="readonly")
        self.cb_col_y.grid(row=0, column=3, padx=10, sticky="w")

        tk.Label(param_frame, text="Số cụm (K):", bg="white").grid(row=1, column=0, sticky="e", pady=5)
        self.ent_k = tk.Entry(param_frame, width=10)
        self.ent_k.insert(0, "3")
        self.ent_k.grid(row=1, column=1, padx=10, sticky="w")

        tk.Label(param_frame, text="Min Support:", bg="white").grid(row=1, column=2, sticky="e")
        self.ent_sup = tk.Entry(param_frame, width=10)
        self.ent_sup.insert(0, "0.5")
        self.ent_sup.grid(row=1, column=3, padx=10, sticky="w")

        tk.Label(param_frame, text="Min Confidence:", bg="white").grid(row=1, column=4, sticky="e")
        self.ent_conf = tk.Entry(param_frame, width=10)
        self.ent_conf.insert(0, "0.7")
        self.ent_conf.grid(row=1, column=5, padx=10, sticky="w")

        btn_grid = tk.Frame(form_frame, bg="white")
        btn_grid.pack(pady=10)
        
        btn_opts = {
            "fg": "white", 
            "bg": "#3498db", 
            "relief": "flat", 
            "width": 28, 
            "pady": 7, 
            "font": ("Arial", 9, "bold"), 
            "cursor": "hand2"
        }
        
        tk.Button(btn_grid, text="Tính Tương Quan Pearson", command=self.run_correlation, **btn_opts).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(btn_grid, text="Phân Cụm (K-Means)", command=self.run_clustering, **btn_opts).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(btn_grid, text="Cây Quyết Định (ID3)", command=self.run_decision_tree, **btn_opts).grid(row=0, column=2, padx=10, pady=5)
        
        tk.Button(btn_grid, text="Naive Bayes (Thường)", command=lambda: self.run_naive(False), **btn_opts).grid(row=1, column=0, padx=10, pady=5)
        tk.Button(btn_grid, text="Naive Bayes (Laplace)", command=lambda: self.run_naive(True), **btn_opts).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(btn_grid, text="Luật Kết Hợp (Apriori)", command=self.run_association, **btn_opts).grid(row=1, column=2, padx=10, pady=5)
        
        tk.Button(btn_grid, text="Phân Tích Tập Thô (Rough Set)", command=self.run_rough_set, bg="#27ae60", fg="white", relief="flat", width=28, pady=7, font=("Arial", 9, "bold")).grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self.scrollable_frame, text="BẢNG KẾT QUẢ PHÂN TÍCH CHI TIẾT", fg="#27ae60", bg="#f4f7f6", font=("Arial", 14, "bold")).pack(pady=10)
        
        res_container = tk.Frame(self.scrollable_frame, bg="white", bd=1, relief="solid", height=320)
        res_container.pack(padx=40, pady=5, fill="x")
        res_container.pack_propagate(False) 

        self.result_area = tk.Text(res_container, font=("Consolas", 10), padx=15, pady=15, bg="white", relief="flat")
        self.result_area.pack(side="left", expand=True, fill="both")
        
        scrollbar = tk.Scrollbar(res_container, command=self.result_area.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_area.config(yscrollcommand=scrollbar.set)

        self.img_label = tk.Label(self.scrollable_frame, bg="#f4f7f6")
        self.img_label.pack(pady=20, fill="both", expand=True)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path, encoding='utf-8')
                self.column_names = self.df.columns.tolist()
                
                self.cb_col_x['values'] = self.column_names
                self.cb_col_y['values'] = self.column_names
                
                self.lbl_file.config(text=f"Đã nạp: {os.path.basename(file_path)}", fg="#27ae60")
                self.clear_display()
                self.result_area.insert(tk.END, "--- DỮ LIỆU ĐẦU VÀO (10 DÒNG ĐẦU) ---\n\n")
                self.result_area.insert(tk.END, self.df.head(10).to_string())
            except Exception as e:
                messagebox.showerror("Lỗi nạp file", str(e))

    def clear_display(self):
        self.result_area.delete('1.0', tk.END)
        self.img_label.config(image="")
        self.photo = None 

    def get_clean_df(self):
        if self.df is None: return None
        df_c = self.df.copy()
        last_val = str(df_c.iloc[-1, 0])
        if "Tong" in last_val or "Tổng" in last_val:
            return df_c.iloc[:-1].copy()
        return df_c

    def run_correlation(self):
        if self.df is None: return
        self.clear_display()
        try:
            cx = self.cb_col_x.get()
            cy = self.cb_col_y.get()
            if not cx or not cy: 
                return messagebox.showwarning("Lỗi", "Chọn đủ 2 cột!")
            
            df_c = self.get_clean_df()
            x = pd.to_numeric(df_c[cx], errors='coerce').dropna().tolist()
            y = pd.to_numeric(df_c[cy], errors='coerce').dropna().tolist()
            
            r, text = calculate_pearson(x, y)
            
            res = "PHÂN TÍCH TƯƠNG QUAN PEARSON\n"
            res += "----------------------------------------\n"
            res += f" - Cột X: {cx}\n"
            res += f" - Cột Y: {cy}\n"
            res += f" - Hệ số tương quan (r): {r}\n"
            res += f" - Nhận xét: {text}"
            self.result_area.insert(tk.END, res)
        except Exception as e: 
            messagebox.showerror("Lỗi", str(e))

    def run_clustering(self):
        if self.df is None: return messagebox.showwarning("Cảnh báo", "Vui lòng nạp dữ liệu!")
        self.clear_display()
        try:
            k_val = int(self.ent_k.get())
            df_c = self.get_clean_df()
            
            df_clustering = df_c.iloc[:, 1:].copy()
            
            le = LabelEncoder()
            for col in df_clustering.columns:
                if df_clustering[col].dtype == type(object) or df_clustering[col].dtype.name == 'category':
                    df_clustering[col] = le.fit_transform(df_clustering[col].astype(str))
            
            data = df_clustering.values
            
            km = KMeansAlgorithm(n_clusters=k_val)
            
            initial_labels, initial_centroids = km.initialize_random_clusters(data)
            
            iters, final_centroids, final_labels = km.fit(data)
            
            res = f"GOM CỤM DỮ LIỆU K-MEANS (K={k_val})\n"
            res += "=" * 50 + "\n\n"
            
            res += "1. TRẠNG THÁI KHỞI TẠO:\n"
            res += f" - Tọa độ tâm cụm ban đầu:\n{initial_centroids}\n\n"
            
            res += "2. QUÁ TRÌNH DỊCH CHUYỂN TÂM CỤM:\n"
            for it in iters:
                res += f" + Vòng lặp thứ {it['vong_lap']}:\n"
                res += f"   -> Tọa độ mới:\n{np.array(it['trong_tam'])}\n"
            
            res += "\n3. KẾT QUẢ HỘI TỤ CUỐI CÙNG:\n"
            res += "-" * 30 + "\n"
            res += f" - Tọa độ các tâm cụm đích:\n{final_centroids}\n\n"
            res += f" - Phân cụm (Nhãn) của từng dòng dữ liệu:\n{final_labels}\n"
            
            self.result_area.insert(tk.END, res)
        except Exception as e: 
            messagebox.showerror("Lỗi K-Means", f"Đã xảy ra lỗi:\n{str(e)}")

    def run_decision_tree(self):
        if self.df is None: 
            return
        self.clear_display()
        try:
            df_clean = self.get_clean_df()
            my_logs = []
            
            tree_dict, calculation_logs = build_tree_with_logs(df_clean, current_node_name="Gốc", logs_list=my_logs)
            
            image_name = "Cay_Quyet_Dinh_ID3.png"
            save_tree_image(tree_dict, image_name)
            
            output_text = "THUẬT TOÁN CÂY QUYẾT ĐỊNH ID3 (ĐỘ LỢI THÔNG TIN - INFORMATION GAIN)\n"
            output_text += "=" * 75 + "\n"
            output_text += "BẢNG PHÂN TÍCH TOÁN HỌC CHI TIẾT THEO GIÁO TRÌNH PDF:\n\n"
            
            output_text += "".join(calculation_logs)
            
            output_text += "CẤU TRÚC ĐỆ QUY DẠNG DICTIONARY NỘI BỘ:\n"
            output_text += str(tree_dict) + "\n\n"
            
            output_text += "TẬP CÁC LUẬT RÚT RA TỪ CÂY QUYẾT ĐỊNH (CLASSIFICATION RULES):\n"
            output_text += "-" * 75 + "\n"
            
            generated_rules = extract_rules(tree_dict)
            
            for idx, rule in enumerate(generated_rules, 1):
                output_text += f" Luật {idx}:{rule}\n"
            output_text += "-" * 75 + "\n\n"
            
            output_text += "MÔ HÌNH ĐỒ THỊ CÂY QUYẾT ĐỊNH TRỰC QUAN:\n"
            output_text += "-" * 75 + "\n"
            
            self.result_area.insert(tk.END, output_text)
            
            if os.path.exists(image_name):
                img = Image.open(image_name)
                img = img.resize((600, 450), Image.Resampling.LANCZOS)
                self.tree_image_embedded = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(self.result_area, image=self.tree_image_embedded, bg=self.result_area.cget('bg'))
                img_label.image = self.tree_image_embedded
                
                self.result_area.window_create(tk.END, window=img_label)
                self.result_area.insert(tk.END, "\n\n--- HOÀN THÀNH XUẤT BÁO CÁO ID3 ---")
                
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Lỗi thực thi ID3", f"Có lỗi xảy ra trong quá trình chạy giải thuật:\n{str(e)}")
    def run_association(self):
        if self.df is None: return
        self.clear_display()
        try:
            sup = float(self.ent_sup.get())
            conf = float(self.ent_conf.get())
            df_c = self.get_clean_df()
            
            if 'id' in str(df_c.columns[0]).lower() or 'tid' in str(df_c.columns[0]).lower() or df_c.iloc[:, 0].dtype == object:
                df_items = df_c.iloc[:, 1:]
            else:
                df_items = df_c

            trans = []
            total_elements = df_items.size
            
            x_count = df_items.applymap(lambda s: str(s).strip().lower() == 'x').sum().sum()

            if x_count > 0 and (x_count / total_elements) > 0.05:
                item_names = df_items.columns.tolist()
                for _, row in df_items.iterrows():
                    transaction = [col for col in item_names if str(row[col]).strip().lower() == 'x']
                    if transaction:
                        trans.append(transaction)
                        
            elif df_items.shape[1] >= 3 and df_items.applymap(lambda s: '=' not in str(s)).all().all():
                for _, row in df_items.iterrows():
                    transaction = [f"{col}={str(row[col]).strip()}" for col in df_items.columns if pd.notna(row[col]) and str(row[col]).strip() not in ['', 'nan', 'None']]
                    if transaction:
                        trans.append(transaction)
            else:
                raw_lists = df_items.applymap(str).values.tolist()
                for r in raw_lists:
                    clean_row = [i for i in r if i not in ['nan', '', 'None', 'NaN']]
                    if clean_row:
                        trans.append(clean_row)

            freq, maximal = apriori(trans, sup)
            rules = generate_rules(freq, conf)
            
            target_col = df_items.columns[-1]
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', None)
            res = "PHÂN TÍCH LUẬT KẾT HỢP \n"
            res += "=" * 65 + "\n"
            res += f" - Ngưỡng min_sup: {sup} | Ngưỡng min_conf: {conf}\n"
            res += f" - Tổng số giao dịch xử lý: {len(trans)}\n\n"

            res += "1. DANH SÁCH TẬP PHỔ BIẾN (FREQUENT ITEMSETS):\n"
            df_freq = pd.DataFrame(freq)
            if not df_freq.empty:
                df_freq = df_freq[['support', 'itemsets']]

                df_freq['itemsets'] = df_freq['itemsets'].apply(lambda x: f"({', '.join(list(x))})")
                res += df_freq.to_string(index=True) + "\n\n"
            else:
                res += " - (Không tìm thấy tập phổ biến nào thỏa mãn ngưỡng dữ liệu)\n\n"

            res += f"2. TẬP PHỔ BIẾN TỐI ĐẠI ({len(maximal)} tập):\n"
            for m in maximal:
                res += f"  ➔ {{ {', '.join(list(m['itemsets']))} }}\n"
            res += "\n"

            res += "3. DANH SÁCH LUẬT KẾT HỢP TÌM THẤY (ASSOCIATION RULES):\n"
            df_rules = pd.DataFrame(rules)
            
            if not df_rules.empty:
                df_rules['antecedents'] = df_rules['antecedents'].apply(lambda x: f"({', '.join(list(x))})")
                df_rules['consequents'] = df_rules['consequents'].apply(lambda x: f"({', '.join(list(x))})")
                df_rules = df_rules[['antecedents', 'consequents', 'support', 'confidence']]
                df_rules['support'] = df_rules['support'].astype(float).round(4)
                df_rules['confidence'] = df_rules['confidence'].astype(float).round(4)
                is_targeted_data = df_rules['consequents'].str.contains(f"{target_col}=").any()
                if is_targeted_data:
                    df_rules['is_target'] = df_rules['consequents'].apply(lambda x: 1 if f"{target_col}=" in x else 0)
                    df_rules = df_rules.sort_values(by='is_target', ascending=False).drop(columns=['is_target'])
                
                res += df_rules.to_string(index=True)
            else:
                res += " - (Không tìm thấy luật kết hợp nào đạt ngưỡng tối thiểu)"
            
            self.result_area.insert(tk.END, res)
            
        except Exception as e: 
            messagebox.showerror("Lỗi Phân Tích Apriori", f"Không thể xử lý dữ liệu file:\n{str(e)}")

    def run_rough_set(self):
        if self.df is None: return
        self.clear_display()
        try:
            df_c = self.get_clean_df()
            approx, reducts, rules = run_rough_set_analysis(df_c)
            
            res = "KẾT QUẢ PHÂN TÍCH TẬP THÔ (ROUGH SETS)\n"
            res += "----------------------------------------\n"
            res += "1. Phân tích xấp xỉ lớp quyết định:\n"
            for a in approx:
                res += f" - Lớp {a['Lớp']}:\n"
                res += f"   + Hạ (Lower): {a['Xấp xỉ dưới']}\n"
                res += f"   + Thượng (Upper): {a['Xấp xỉ trên']}\n"
                res += f"   + Độ chính xác: {a['Độ chính xác']}\n"
            
            res += f"\n2. Các tập rút gọn (Reducts): {reducts}\n"
            res += f"\n3. Các luật trích xuất từ tập rút gọn:\n"
            for rule in rules:
                res += f" + {rule}\n"
            
            self.result_area.insert(tk.END, res)
        except Exception as e: 
            messagebox.showerror("Lỗi", str(e))
    def run_naive(self, smooth):
        if self.df is None: 
            return
        self.clear_display()
        try:
            df_c = self.get_clean_df()
            
            feature_names = list(df_c.columns[1:-1])
            target_name = df_c.columns[-1]
            
            X_train = df_c.iloc[:, 1:-1].values
            y_train = df_c.iloc[:, -1].values
            
            nb = NaiveBayesClassifier(smoothing=smooth)
            nb.fit(X_train, y_train)
            
            _, detailed_logs = nb.predict_with_detailed_logging(X_train, feature_names, target_name)
            
            self.result_area.insert(tk.END, detailed_logs)
            
        except Exception as e: 
            messagebox.showerror("Lỗi Naive Bayes", f"Đã xảy ra lỗi hệ thống:\n{str(e)}")
if __name__ == "__main__":
    root = tk.Tk()
    app = DataMiningApp(root)
    root.mainloop()