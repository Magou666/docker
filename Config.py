# coding:utf-8

# 图片及视频检测结果保存路径
save_path = r'save_data'

# 使用的模型路径
model_path = r'runs\detect\strawberry_model7\weights\best.pt'
# 数据集类别与名称
names = {
    0: 'xiaomai_bacteria_leaf_banbing',
    1: 'xiaomai_sui_bing',
    2: 'xiaomai_ye_xiubing',
    3: 'xiaomai_song_bibing',
    4: 'xiaomai_baifenbing',
    5: 'xiaomai_yebanbing_chimeibing',
    6: 'xiaomai_jingxiubing',
    7: 'xiaomai_tiaoxiubing'
}

# 数据集类别中文
CH_names = {
    0: '小麦细菌性叶斑病（黑秆病）',
    1: '小麦穗病（麦穗霉病）',
    2: '小麦叶锈病',
    3: '小麦松秕病',
    4: '小麦白粉病',
    5: '小麦叶斑病（赤霉病）',
    6: '小麦茎锈病',
    7: '小麦条锈病'
}

# 原始类别对应关系（用于向后兼容）
# 如果需要恢复旧模型，可以使用这个映射
old_names = {0: 'strain', 1: 'flower', 3: "berry"}
old_CH_names = ['枝梗', '花', '果子']

# csv文件保存路径
csv_save_path = r'save_data/save_detect_data.csv';

# 像素到毫米的转换比例 (定义每毫米多少像素)
# 如根据实际测量情况调整此值
pixels_per_mm = 10.0  # 默认值：10像素 = 1毫米