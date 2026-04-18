import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

print("正在寻找 logfoldchange 数据文件...")
# 💡 提示：如果你想画 2002 或 3357 模型的结果，只需把这里的 919 改掉即可！
file_list = glob.glob('*3357feature*logfoldchange.csv')

if len(file_list) == 0:
    print("没找到 csv 文件，请检查文件名！")
else:
    file_name = file_list[0]
    print(f"正在读取文件: {file_name}")
    
    df = pd.read_csv(file_name)
    variant_names = df.iloc[:, :5].astype(str).agg('_'.join, axis=1)
    df_scores = df.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')
    df_scores.index = variant_names
    
    # 极其重要：填补可能存在的空值，否则底层的 scipy 聚类算法会直接崩溃
    df_scores.fillna(0, inplace=True)
    
    # 为了展现聚类的宏观震撼效果，我们把截取范围扩大到 50 个突变，100 个特征
    subset = df_scores.iloc[:50, :100]
    
    print("数据提取成功，正在进行复杂的层次聚类并绘制高颜值热图...")
    
    # 使用 clustermap 替代 heatmap，这是顶刊标配！
    g = sns.clustermap(subset, 
                       cmap='RdBu_r', 
                       center=0,
                       figsize=(16, 12),
                       linewidths=0.5, # 加上极其细微的网格线，让方块更有质感
                       cbar_kws={'label': 'Log Fold Change'},
                       tree_kws={'linewidths': 1.5}) # 加粗边缘的聚类树线条
    
    # 调整标题和标签（clustermap 的图层结构较复杂，需特殊处理）
    feature_name = file_name.split("_")[-1].split(".")[0]
    g.fig.suptitle(f'Variant Effect Prediction (Clustered) - {feature_name}', fontsize=20, y=1.02)
    g.ax_heatmap.set_xlabel('Chromatin Features (Clustered)', fontsize=14)
    g.ax_heatmap.set_ylabel('Variants (Clustered)', fontsize=14)
    
    # 保存为超高清 PNG
    output_png = 'Figure3_Clustered_Heatmap.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"🎉 顶刊级别聚类热图已生成: {output_png}")