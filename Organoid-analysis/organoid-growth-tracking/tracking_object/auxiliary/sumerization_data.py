# import os
# import pandas as pd
#
# # --- 配置区域 ---
# # 存放那些 VID1495_C2_1 等文件夹的父目录
# parent_folder = '/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round1'
# # 输出汇总文件的名称
# output_file = '/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round1_summary_traces_area.xlsx'
# # 目标文件名
# target_filename = 'traces_area_matrix.xlsx'
# # 需要匹配的后缀范围 (C2_1 到 C2_11)
# target_suffixes = [f'C2_{i}' for i in range(1, 12)]
# # ----------------
#
# all_data = []
#
# # 遍历父文件夹下的所有子文件夹
# for folder_name in os.listdir(parent_folder):
#     folder_path = os.path.join(parent_folder, folder_name)
#
#     # 检查是否为文件夹，并且后缀是否在我们的目标范围内
#     if os.path.isdir(folder_path):
#         # 提取后缀（例如从 tracking_result_VID1495_C2_1 提取 C2_1）
#         suffix = folder_name.split('_')[-2] + '_' + folder_name.split('_')[-1] if '_' in folder_name else ""
#
#         # 简单检查：如果文件夹名以指定的后缀结尾
#         if any(folder_name.endswith(s) for s in target_suffixes):
#             file_path = os.path.join(folder_path, target_filename)
#
#             if os.path.exists(file_path):
#                 print(f"正在读取: {folder_name}...")
#                 # 读取 Excel
#                 df = pd.read_excel(file_path)
#
#                 # 可选：添加一列用来标记数据来源哪个文件夹
#                 df['source_folder'] = folder_name
#
#                 all_data.append(df)
#             else:
#                 print(f"跳过: {folder_name} (未找到 {target_filename})")
#
# # 合并所有数据
# if all_data:
#     final_df = pd.concat(all_data, ignore_index=True)
#     # 保存到 Excel
#     final_df.to_excel(output_file, index=False)
#     print(f"\n--- 汇总完成！已保存至 {output_file} ---")
# else:
#     print("未发现匹配的数据，请检查路径和后缀设置。")

# import os
# import re
# import pandas as pd
#
# # =========================
# # 配置区（改这里）
# # =========================
# parent_folder = "/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round3"
# output_file   = "/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round3_IDI1_g2.xlsx"
# target_filename = "traces_area_matrix.xlsx"
#
# # 匹配 C2_x 到 C11_x
# # folder_pattern = re.compile(r"H([2-9]|1[0-1])_\d+")
# folder_pattern = re.compile(r"([A-H])11_(\d+)", re.IGNORECASE)
#
#
#
#
# # 是否给每个文件的非第一列加前缀，避免 idx1/idx2 重名覆盖
# ADD_PREFIX = True
# # =========================
#
# # def sort_key(name):
# #     m = folder_pattern.search(name)
# #     c_num = int(m.group(1))
# #     x_num = int(name.split("_")[-1])
# #     return (c_num, x_num)
#
# def sort_key(name: str):
#     m = folder_pattern.search(name)
#     letter = m.group(1).upper()          # 'A'~'H'
#     x_num  = int(m.group(2))             # 1,2,3...
#     letter_idx = ord(letter) - ord('A')  # A=0, B=1, ..., H=7
#     return (letter_idx, x_num, name.lower())
#
#
# def list_folders_sorted(parent: str):
#     folders = []
#     for name in os.listdir(parent):
#         if folder_pattern.fullmatch(name) or folder_pattern.search(name):
#             folders.append(name)
#     folders.sort(key=sort_key)
#     return folders
#
#
# def read_one_excel(file_path: str):
#     df = pd.read_excel(file_path)
#     if df.shape[1] < 1:
#         return None
#
#     # 以“第一列”为key列（例如 frame_idx）
#     key_col = df.columns[0]
#     df = df.copy()
#
#     # 清理 key 列：转数值（如果是数字字符串），保留 NaN
#     df[key_col] = pd.to_numeric(df[key_col], errors="coerce")
#
#     return df, key_col
#
#
# folders = list_folders_sorted(parent_folder)
# print(f"[INFO] matched folders: {len(folders)}")
#
# merged = None
# global_key_col = None
#
# for i, folder_name in enumerate(folders, start=1):
#     file_path = os.path.join(parent_folder, folder_name, target_filename)
#     if not os.path.exists(file_path):
#         print(f"[WARN] missing file: {file_path}")
#         continue
#
#     try:
#         print(f"[INFO] ({i}/{len(folders)}) reading: {folder_name}")
#         df, key_col = read_one_excel(file_path)
#         if df is None:
#             print(f"[WARN] empty: {file_path}")
#             continue
#
#         # 第一次确定全局key列名（一般就是 frame_idx）
#         if global_key_col is None:
#             global_key_col = key_col
#         else:
#             # 如果后面文件第一列名不同，也强制对齐到 global_key_col
#             if key_col != global_key_col:
#                 df = df.rename(columns={key_col: global_key_col})
#                 key_col = global_key_col
#
#         # 只保留：第一列 + 其余列（全部保留）
#         # 将 key 列设为 index，方便 outer join 合并（=“把所有第一列合并”）
#         df = df.set_index(key_col)
#
#         # 给非第一列加前缀，避免不同文件 idx1/idx2 重名
#         if ADD_PREFIX:
#             df = df.rename(columns={c: f"{folder_name}__{c}" for c in df.columns})
#
#         if merged is None:
#             merged = df
#         else:
#             merged = merged.join(df, how="outer")
#
#     except Exception as e:
#         print(f"[ERROR] failed on {folder_name}: {e}")
#
# if merged is None:
#     raise RuntimeError("没有读到任何可用文件，无法合并。")
#
# # index（frame_idx）排序、恢复为第一列
# merged = merged.sort_index()
# merged = merged.reset_index().rename(columns={merged.columns[0]: global_key_col})
#
# # 输出
# merged.to_excel(output_file, index=False)
# print(f"\n[DONE] saved -> {output_file}")
# print(f"[INFO] rows={merged.shape[0]}, cols={merged.shape[1]}")


import os
import re
import pandas as pd

# =========================
# 配置区（改这里）
# =========================
parent_folder = "/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round5"
output_file   = "/Volumes/Expansion/Droplets project images analysis for yimin-xyn/CRISPR Validation/crisper_valid/parsed_output/round5_output/round_5_UBE2V1_g2.xlsx"
target_filename = "traces_area_matrix.xlsx"

# 从文件夹名中匹配 A3_1 ~ H3_999（在任意位置）
tag_pattern = re.compile(r"([A-H])8_(\d+)", re.IGNORECASE)

# folder_pattern = re.compile(r"H([2-9]|1[0-1])_\d+")

# 列名前缀用哪个：
#   True  -> 用完整文件夹名，如 tracking_result_VID1495_A3_1__idx1
#   False -> 只用 A3_1 这种 tag，如 A3_1__idx1
PREFIX_USE_FULL_FOLDERNAME = True
# =========================


# def sort_key(name):
#     m = folder_pattern.search(name)
#     c_num = int(m.group(1))
#     x_num = int(name.split("_")[-1])
#     return (c_num, x_num)

def list_folders_sorted(parent: str):
    folders = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        if tag_pattern.search(name):
            folders.append(name)

    def sort_key(folder_name: str):
        m = tag_pattern.search(folder_name)
        letter = m.group(1).upper()           # A~H
        num = int(m.group(2))                # 1,2,3...
        return (ord(letter) - ord('A'), num, folder_name.lower())

    folders.sort(key=sort_key)
    return folders


def read_and_standardize(file_path: str):
    """
    读单个 traces_area_matrix.xlsx
    假设格式为：第一列 frame_idx，其余列为 idx1, idx2, ...
    但为了稳，按“列位置”强制改名为 frame_idx + idx1..idxN
    """
    df = pd.read_excel(file_path, engine="openpyxl")
    if df.shape[1] < 1:
        return None

    df = df.copy()

    # 第一列作为 frame_idx（不信任列名）
    frame_col = df.columns[0]
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")

    # 强制列名：frame_idx, idx1..idxN
    n_other = df.shape[1] - 1
    new_cols = ["frame_idx"] + [f"idx{i}" for i in range(1, n_other + 1)]
    df.columns = new_cols

    return df


folders = list_folders_sorted(parent_folder)
print(f"[INFO] matched folders: {len(folders)}")
if not folders:
    raise RuntimeError("没有匹配到任何文件夹，请检查 parent_folder 或命名是否包含 A3_1 这类标记。")

merged = None

for i, folder_name in enumerate(folders, start=1):
    file_path = os.path.join(parent_folder, folder_name, target_filename)
    if not os.path.exists(file_path):
        print(f"[WARN] missing file: {file_path}")
        continue

    print(f"[INFO] ({i}/{len(folders)}) reading: {folder_name}")
    df = read_and_standardize(file_path)
    if df is None:
        print(f"[WARN] empty excel: {file_path}")
        continue

    # 取前缀
    if PREFIX_USE_FULL_FOLDERNAME:
        prefix = folder_name
    else:
        m = tag_pattern.search(folder_name)
        prefix = f"{m.group(1).upper()}3_{int(m.group(2))}"  # e.g. A3_1

    # 设置 index 仅用 frame_idx 合并
    df = df.set_index("frame_idx")

    # 给所有 idx 列加前缀
    df = df.rename(columns={c: f"{prefix}__{c}" for c in df.columns})

    if merged is None:
        merged = df
    else:
        merged = merged.join(df, how="outer")

if merged is None:
    raise RuntimeError("没有读到任何可用 excel 文件，无法合并。")

# 排序并输出
merged = merged.sort_index()
merged = merged.reset_index()  # index -> frame_idx 列

merged.to_excel(output_file, index=False)
print(f"\n[DONE] saved -> {output_file}")
print(f"[INFO] rows={merged.shape[0]}, cols={merged.shape[1]}")
