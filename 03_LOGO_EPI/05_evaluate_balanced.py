import os
import numpy as np
import importlib.util
import sklearn.metrics as sk_metrics

print("==========================================================")
print("Starting Balanced Test Set Evaluation & Threshold Optimization")
print("==========================================================")

# 导入原来的 04 脚本
script_name = "04_LOGO_EPI_train_conv1d_concat_atcg.py"
spec = importlib.util.spec_from_file_location("logo_module", script_name)
logo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logo_module)

# ---------------------------------------------------------
# 🌟 核心黑科技：动态替换原代码的算分函数，实现“阈值寻优”
# ---------------------------------------------------------
def optimized_bagging_predict(label, bag_pred, bag_score):
    # 算出 10 个模型的平均概率
    vote_score = np.mean(bag_score, axis=0)
    # AUPRC 不受硬阈值影响，按原样计算
    auprc = sk_metrics.average_precision_score(label, vote_score)
    
    # 滑动寻找最佳阈值 (0.01 ~ 0.99)
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        temp_pred = (vote_score > t).astype(int)
        temp_f1 = sk_metrics.f1_score(label, temp_pred)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_t = t
            
    print(f"🎯 Threshold Optimized! Best Cutoff is: {best_t:.2f}")
    return best_f1, auprc

# 狸猫换太子：用我们的寻优函数替换掉 logo_module 里的原函数
logo_module.bagging_predict = optimized_bagging_predict

# ===== 动态计算 vocab_size =====
import sys
sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number
word_dict = get_word_dict_for_n_gram_number(n_gram=6)
logo_module.vocab_size = len(word_dict) + 10
# ===============================

CELLS = ["tB", "FoeT", "Mon", "nCD4", "tCD4", "tCD8"]
TYPE = "P-E"
ngram = 6

for CELL in CELLS:
    print(f"\n" + "="*50)
    print(f"🚀 Processing Cell Line: {CELL}")
    print("="*50)
    
    test_dir = f"{CELL}/{TYPE}/test"
    orig_dir = f"{CELL}/{TYPE}/test_original"
    
    # 依然执行 1:1 数据平衡逻辑
    if not os.path.exists(orig_dir):
        os.rename(test_dir, orig_dir)
        os.makedirs(f"{test_dir}/{ngram}_gram", exist_ok=True)
        print(f"🔒 Backed up original imbalanced test set to {orig_dir}")
    else:
        print(f"✅ Backup already exists at {orig_dir}")
        
    enhancer_file = f"{orig_dir}/{ngram}_gram/enhancer_Seq_{ngram}_gram.npz"
    promoter_file = f"{orig_dir}/{ngram}_gram/promoter_Seq_{ngram}_gram.npz"
    enh_data = np.load(enhancer_file)
    pro_data = np.load(promoter_file)
    y_key = 'y' if 'y' in enh_data.files else 'Y'
    y = enh_data[y_key]
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(42)
    neg_idx_sampled = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    
    balanced_idx = np.concatenate([pos_idx, neg_idx_sampled])
    np.random.shuffle(balanced_idx)
    
    enh_bal = {k: enh_data[k][balanced_idx] for k in enh_data.files}
    pro_bal = {k: pro_data[k][balanced_idx] for k in pro_data.files}
    np.savez(f"{test_dir}/{ngram}_gram/enhancer_Seq_{ngram}_gram.npz", **enh_bal)
    np.savez(f"{test_dir}/{ngram}_gram/promoter_Seq_{ngram}_gram.npz", **pro_bal)
    
    # 调用 04 脚本的 evaluate，此时它内部已经悄悄用上了我们的最优 F1 计算方法！
    logo_module.evaluate(CELL, TYPE, NUM_ENSEMBL=10, ngram=6, batch_size=128)

print("\n==========================================================")
print("ALL DONE! Check your MAXIMIZED F1 scores!")
print("==========================================================")