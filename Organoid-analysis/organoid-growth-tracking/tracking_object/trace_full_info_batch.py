#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set

import numpy as np
import pandas as pd


# -------------------------
# Utils
# -------------------------
def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def safe_int(x, default=None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default


def parse_parent_ids(val) -> List[int]:
    """
    parent_ids 可能是：
      - FALSE / TRUE / 空
      - "48,51"
      - "5"
      - 数字/浮点(5, 5.0)
    统一转成 List[int]
    """
    if val is None or pd.isna(val):
        return []
    s = str(val).strip()
    if s == "" or s.upper() == "FALSE" or s.upper() == "TRUE" or s.lower() == "nan":
        return []
    parts = re.split(r"[,\s;]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(float(p)))
        except Exception:
            pass
    return out


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_excel_file(p: str) -> bool:
    if not os.path.isfile(p):
        return False
    name = os.path.basename(p)
    if name.startswith("~$"):  # Excel 临时锁文件
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in [".xlsx", ".xls", ".xlsm", ".xlsb"]


def sanitize_relpath_as_folder(relpath: str) -> str:
    """
    用于 recursive 模式下：把相对路径变成安全的文件夹名
    e.g. "sub/a.xlsx" -> "sub__a"
    """
    relpath = relpath.replace("\\", "/")
    rel_no_ext = os.path.splitext(relpath)[0]
    # 替换路径分隔符与非法字符
    s = re.sub(r"[\/]+", "__", rel_no_ext)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-_\.]+", "_", s)
    return s.strip("_") or "output"


# -------------------------
# Core tracing
# -------------------------
@dataclass
class TraceRow:
    frame_idx: int
    object_id: int
    mask_area: float
    timestamp: Optional[str] = None
    parent_ids_raw: Optional[str] = None  # 便于调试/记录


def build_frame_index(df: pd.DataFrame, frame_col: str, obj_col: str) -> Dict[int, Dict[int, List[int]]]:
    """
    frame_idx -> object_id -> row_index list
    """
    idx: Dict[int, Dict[int, List[int]]] = {}
    for i, r in df.iterrows():
        f = safe_int(r.get(frame_col), None)
        o = safe_int(r.get(obj_col), None)
        if f is None or o is None:
            continue
        idx.setdefault(f, {}).setdefault(o, []).append(i)
    return idx


def pick_best_row(df: pd.DataFrame, row_indices: List[int], area_col: str) -> int:
    """同一候选多行，选 mask_area 最大的行"""
    if len(row_indices) == 1:
        return row_indices[0]
    areas = []
    for i in row_indices:
        areas.append(safe_float(df.at[i, area_col], np.nan))
    k = int(np.nanargmax(np.array(areas)))
    return row_indices[k]


def find_parent_row_with_skip(
    df: pd.DataFrame,
    frame_map: Dict[int, Dict[int, List[int]]],
    cur_frame: int,
    parent_ids: List[int],
    area_col: str,
    max_skip: int,
) -> Optional[Tuple[int, int, int]]:
    """
    在 cur_frame-1 ... cur_frame-max_skip 里找 parent。
    返回 (parent_frame, parent_row_index, chosen_parent_id)
    规则：
      - 从近到远扫描帧
      - 同一帧内 parent_ids 多个：选 mask_area 最大的那个 parent 行
    """
    for pf in range(cur_frame - 1, cur_frame - max_skip - 1, -1):
        if pf not in frame_map:
            continue
        best_row = None
        best_area = -np.inf
        best_pid = None

        for pid in parent_ids:
            rows = frame_map[pf].get(pid, [])
            if not rows:
                continue
            ridx = pick_best_row(df, rows, area_col=area_col)
            area = safe_float(df.at[ridx, area_col], np.nan)
            if np.isnan(area):
                continue
            if area > best_area:
                best_area = area
                best_row = ridx
                best_pid = pid

        if best_row is not None:
            return pf, best_row, int(best_pid)

    return None


def trace_back_one(
    df: pd.DataFrame,
    frame_map: Dict[int, Dict[int, List[int]]],
    start_frame: int,
    start_oid: int,
    frame_col: str,
    obj_col: str,
    area_col: str,
    parent_col: str,
    ts_col: Optional[str],
    max_skip: int = 3,
    max_steps: int = 100000,
) -> Tuple[List[TraceRow], List[Tuple[int, int, List[int]]]]:
    """
    从 (start_frame, start_oid) 开始向前追溯主链。
    返回：
      - trace: earliest->latest
      - merges: List[(frame_idx, object_id, parent_ids_list)]  记录发生“合并”的节点（parent_ids>1）
    """
    rows = frame_map.get(start_frame, {}).get(start_oid, [])
    if not rows:
        raise ValueError(f"start (frame={start_frame}, oid={start_oid}) not found in input.")
    cur_idx = pick_best_row(df, rows, area_col=area_col)

    trace_latest_to_earliest: List[TraceRow] = []
    merges: List[Tuple[int, int, List[int]]] = []

    steps = 0
    while cur_idx is not None and steps < max_steps:
        steps += 1
        r = df.loc[cur_idx]
        cur_frame = safe_int(r.get(frame_col), None)
        cur_oid = safe_int(r.get(obj_col), None)
        if cur_frame is None or cur_oid is None:
            break

        parents_raw = None if parent_col not in df.columns else r.get(parent_col)
        parents_list = parse_parent_ids(parents_raw)

        if len(parents_list) > 1:
            merges.append((int(cur_frame), int(cur_oid), parents_list))

        trace_latest_to_earliest.append(
            TraceRow(
                frame_idx=int(cur_frame),
                object_id=int(cur_oid),
                mask_area=safe_float(r.get(area_col), np.nan),
                timestamp=(str(r.get(ts_col)) if ts_col and ts_col in df.columns and not pd.isna(r.get(ts_col)) else None),
                parent_ids_raw=(None if parents_raw is None or pd.isna(parents_raw) else str(parents_raw)),
            )
        )

        if not parents_list:
            break

        found = find_parent_row_with_skip(
            df=df,
            frame_map=frame_map,
            cur_frame=int(cur_frame),
            parent_ids=parents_list,
            area_col=area_col,
            max_skip=max_skip,
        )
        if found is None:
            break

        _, parent_row, _chosen_pid = found
        cur_idx = parent_row

    trace = list(reversed(trace_latest_to_earliest))
    return trace, merges


def trace_back_from_node(
    df: pd.DataFrame,
    frame_map: Dict[int, Dict[int, List[int]]],
    start_frame: int,
    start_oid: int,
    frame_col: str,
    obj_col: str,
    area_col: str,
    parent_col: str,
    ts_col: Optional[str],
    max_skip: int,
    max_steps: int,
    visited: Optional[Set[Tuple[int, int]]] = None,
) -> List[TraceRow]:
    """
    给“合并前的某个 parent 目标”追一条链（不展开多父节点，仍用主链规则：多父取最大面积）。
    visited 用于防循环。
    返回 earliest->latest（相对这个parent分支）
    """
    if visited is None:
        visited = set()

    rows = frame_map.get(start_frame, {}).get(start_oid, [])
    if not rows:
        return []
    cur_idx = pick_best_row(df, rows, area_col=area_col)

    trace_latest_to_earliest: List[TraceRow] = []
    steps = 0
    while cur_idx is not None and steps < max_steps:
        steps += 1
        r = df.loc[cur_idx]
        cur_frame = safe_int(r.get(frame_col), None)
        cur_oid = safe_int(r.get(obj_col), None)
        if cur_frame is None or cur_oid is None:
            break

        key = (int(cur_frame), int(cur_oid))
        if key in visited:
            break
        visited.add(key)

        parents_raw = r.get(parent_col) if parent_col in df.columns else None
        parents_list = parse_parent_ids(parents_raw)

        trace_latest_to_earliest.append(
            TraceRow(
                frame_idx=int(cur_frame),
                object_id=int(cur_oid),
                mask_area=safe_float(r.get(area_col), np.nan),
                timestamp=(str(r.get(ts_col)) if ts_col and ts_col in df.columns and not pd.isna(r.get(ts_col)) else None),
                parent_ids_raw=(None if parents_raw is None or pd.isna(parents_raw) else str(parents_raw)),
            )
        )

        if not parents_list:
            break

        found = find_parent_row_with_skip(
            df=df,
            frame_map=frame_map,
            cur_frame=int(cur_frame),
            parent_ids=parents_list,
            area_col=area_col,
            max_skip=max_skip,
        )
        if found is None:
            break

        _, parent_row, _chosen_pid = found
        cur_idx = parent_row

    return list(reversed(trace_latest_to_earliest))


# -------------------------
# Export helpers
# -------------------------
def traces_to_long_df(traces: List[List[TraceRow]]) -> pd.DataFrame:
    rows = []
    for k, tr in enumerate(traces, start=1):
        for rr in tr:
            rows.append(
                {
                    "idx": k,  # 1..n
                    "frame_idx": rr.frame_idx,
                    "object_id": rr.object_id,
                    "mask_area": rr.mask_area,
                    "timestamp": rr.timestamp,
                    "parent_ids_raw": rr.parent_ids_raw,
                }
            )
    return pd.DataFrame(rows)


def long_df_to_matrix_area(df_long: pd.DataFrame) -> pd.DataFrame:
    mat = df_long.pivot_table(
        index="frame_idx",
        columns="idx",
        values="mask_area",
        aggfunc="first",
    ).sort_index()
    mat.columns = [f"idx{int(c)}" for c in mat.columns]
    mat.insert(0, "frame_idx", mat.index.astype(int))
    mat = mat.reset_index(drop=True)
    return mat


# -------------------------
# IO
# -------------------------
def load_tracking_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet if sheet else 0, engine=None)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -------------------------
# Merge branches extraction
# -------------------------
def build_merge_branches(
    df: pd.DataFrame,
    frame_map: Dict[int, Dict[int, List[int]]],
    idx_id: int,
    merges: List[Tuple[int, int, List[int]]],
    frame_col: str,
    obj_col: str,
    area_col: str,
    parent_col: str,
    ts_col: Optional[str],
    max_skip: int,
    max_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对某个 idx 的所有 merge 节点：
      - merge_id = 1..M
      - 对每个 parent_id，在 (merge_frame-1..merge_frame-max_skip) 找到该 parent 的行（同一帧多行取最大面积）
      - 从该 parent 的 (parent_frame, parent_id) 向前追一条分支轨迹（多父仍取最大面积主链）
    输出：
      - long: columns=[idx, merge_id, branch_parent_id, frame_idx, object_id, mask_area, timestamp]
      - matrix: frame_idx 行，列为 f"idx{idx}_m{merge_id}_p{parent_id}"
    """
    long_rows = []

    for merge_id, (merge_frame, merge_oid, parent_ids) in enumerate(merges, start=1):
        for pid in parent_ids:
            found = find_parent_row_with_skip(
                df=df,
                frame_map=frame_map,
                cur_frame=int(merge_frame),
                parent_ids=[int(pid)],
                area_col=area_col,
                max_skip=max_skip,
            )
            if found is None:
                continue
            pframe, _prow, _ = found
            branch = trace_back_from_node(
                df=df,
                frame_map=frame_map,
                start_frame=int(pframe),
                start_oid=int(pid),
                frame_col=frame_col,
                obj_col=obj_col,
                area_col=area_col,
                parent_col=parent_col,
                ts_col=ts_col,
                max_skip=max_skip,
                max_steps=max_steps,
                visited=set(),
            )
            colname = f"idx{idx_id}_m{merge_id}_p{int(pid)}"
            for rr in branch:
                long_rows.append(
                    {
                        "idx": idx_id,
                        "merge_id": merge_id,
                        "merge_frame": int(merge_frame),
                        "merge_object_id": int(merge_oid),
                        "branch_parent_id": int(pid),
                        "branch_col": colname,
                        "frame_idx": rr.frame_idx,
                        "object_id": rr.object_id,
                        "mask_area": rr.mask_area,
                        "timestamp": rr.timestamp,
                    }
                )

    if not long_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_long = pd.DataFrame(long_rows)

    df_mat = df_long.pivot_table(
        index="frame_idx",
        columns="branch_col",
        values="mask_area",
        aggfunc="first",
    ).sort_index()
    df_mat.insert(0, "frame_idx", df_mat.index.astype(int))
    df_mat = df_mat.reset_index(drop=True)
    return df_long, df_mat


# -------------------------
# Summary
# -------------------------
def build_summary_df(traces: List[List[TraceRow]], merges_list: List[List[Tuple[int, int, List[int]]]]) -> pd.DataFrame:
    rows = []
    for i, (tr, merges) in enumerate(zip(traces, merges_list), start=1):
        if not tr:
            continue
        start = tr[0]
        end = tr[-1]
        rows.append(
            {
                "idx": i,
                "start_frame": int(start.frame_idx),
                "end_frame": int(end.frame_idx),
                "start_object_id": int(start.object_id),
                "end_object_id": int(end.object_id),
                "start_area": float(start.mask_area) if not np.isnan(start.mask_area) else np.nan,
                "end_area": float(end.mask_area) if not np.isnan(end.mask_area) else np.nan,
                "length_frames": int(len(tr)),
                "num_merges": int(len(merges)),
                "merge_parent_counts": ",".join(str(len(m[2])) for m in merges) if merges else "",
            }
        )
    return pd.DataFrame(rows)


# -------------------------
# Batch helpers
# -------------------------
def iter_excel_files(input_path: str, recursive: bool) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path] if is_excel_file(input_path) else []
    if not os.path.isdir(input_path):
        return []
    files = []
    if recursive:
        for root, _dirs, fnames in os.walk(input_path):
            for fn in fnames:
                p = os.path.join(root, fn)
                if is_excel_file(p):
                    files.append(p)
    else:
        for fn in os.listdir(input_path):
            p = os.path.join(input_path, fn)
            if is_excel_file(p):
                files.append(p)
    files.sort()
    return files


def process_one_excel(args, input_file: str, out_dir_for_this_file: str) -> Dict[str, str]:
    """
    对单个 excel 执行原 main 逻辑，输出到 out_dir_for_this_file
    返回写出的关键文件路径字典
    """
    ensure_outdir(out_dir_for_this_file)

    df = load_tracking_excel(input_file, sheet=args.sheet)

    # 基础检查
    for c in [args.frame_col, args.obj_col, args.area_col, args.parent_col]:
        if c not in df.columns:
            raise RuntimeError(f"[{os.path.basename(input_file)}] Missing required column: {c}. Existing columns: {list(df.columns)}")

    ts_col = args.ts_col if args.ts_col in df.columns else None
    frame_map = build_frame_index(df, frame_col=args.frame_col, obj_col=args.obj_col)

    traces: List[List[TraceRow]] = []
    merges_list: List[List[Tuple[int, int, List[int]]]] = []

    if args.mode == "single":
        if args.start_frame is None or args.start_oid is None:
            raise RuntimeError("--mode single requires --start_frame and --start_oid")
        tr, merges = trace_back_one(
            df=df,
            frame_map=frame_map,
            start_frame=args.start_frame,
            start_oid=args.start_oid,
            frame_col=args.frame_col,
            obj_col=args.obj_col,
            area_col=args.area_col,
            parent_col=args.parent_col,
            ts_col=ts_col,
            max_skip=args.max_skip,
            max_steps=args.max_steps,
        )
        traces.append(tr)
        merges_list.append(merges)
    else:
        all_frames = list(frame_map.keys())
        if not all_frames:
            raise RuntimeError("No valid rows with frame/object parsed.")
        last_frame = int(max(all_frames))
        oids = sorted(frame_map[last_frame].keys())
        for oid in oids:
            try:
                tr, merges = trace_back_one(
                    df=df,
                    frame_map=frame_map,
                    start_frame=last_frame,
                    start_oid=int(oid),
                    frame_col=args.frame_col,
                    obj_col=args.obj_col,
                    area_col=args.area_col,
                    parent_col=args.parent_col,
                    ts_col=ts_col,
                    max_skip=args.max_skip,
                    max_steps=args.max_steps,
                )
                traces.append(tr)
                merges_list.append(merges)
            except Exception:
                continue

    # 1) 主链输出
    df_long = traces_to_long_df(traces)
    out_long = os.path.join(out_dir_for_this_file, "traces_long.xlsx")
    df_long.to_excel(out_long, index=False)

    df_mat = long_df_to_matrix_area(df_long)
    out_mat = os.path.join(out_dir_for_this_file, "traces_area_matrix.xlsx")
    df_mat.to_excel(out_mat, index=False)

    # 2) 汇总输出
    df_sum = build_summary_df(traces, merges_list)
    out_sum = os.path.join(out_dir_for_this_file, "objects_summary.xlsx")
    df_sum.to_excel(out_sum, index=False)

    outputs = {
        "traces_long": out_long,
        "traces_area_matrix": out_mat,
        "objects_summary": out_sum,
    }

    # 3) 合并分支输出
    if args.emit_merge_excel:
        all_merge_long = []
        all_merge_mat_parts = []

        for idx_id, merges in enumerate(merges_list, start=1):
            if not merges:
                continue
            mlong, mmat = build_merge_branches(
                df=df,
                frame_map=frame_map,
                idx_id=idx_id,
                merges=merges,
                frame_col=args.frame_col,
                obj_col=args.obj_col,
                area_col=args.area_col,
                parent_col=args.parent_col,
                ts_col=ts_col,
                max_skip=args.max_skip,
                max_steps=args.max_steps,
            )
            if not mlong.empty:
                all_merge_long.append(mlong)
            if not mmat.empty:
                all_merge_mat_parts.append(mmat.set_index("frame_idx"))

        out_merge = os.path.join(out_dir_for_this_file, "objects_merge_branches.xlsx")
        with pd.ExcelWriter(out_merge, engine="openpyxl") as w:
            if all_merge_long:
                pd.concat(all_merge_long, ignore_index=True).to_excel(w, sheet_name="merge_branches_long", index=False)
            else:
                pd.DataFrame().to_excel(w, sheet_name="merge_branches_long", index=False)

            if all_merge_mat_parts:
                merged_mat = pd.concat(all_merge_mat_parts, axis=1).sort_index()
                merged_mat.insert(0, "frame_idx", merged_mat.index.astype(int))
                merged_mat.reset_index(drop=True).to_excel(w, sheet_name="merge_branches_matrix", index=False)
            else:
                pd.DataFrame().to_excel(w, sheet_name="merge_branches_matrix", index=False)

        outputs["objects_merge_branches"] = out_merge

    return outputs


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Trace tracking targets backward using parent_ids with frame skipping.")
    ap.add_argument("-i", "--input", required=True, help="Input tracking Excel file (.xlsx) OR a folder containing excels.")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory (batch outputs go into subfolders).")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet).")

    ap.add_argument("--frame_col", default="frame_idx", help="Frame index column name.")
    ap.add_argument("--obj_col", default="object_id", help="Object id column name.")
    ap.add_argument("--area_col", default="mask_area", help="Area column name.")
    ap.add_argument("--parent_col", default="parent_ids", help="Parent ids column name.")
    ap.add_argument("--ts_col", default="timestamp", help="Timestamp column name (optional).")

    ap.add_argument("--start_frame", type=int, default=None, help="Start tracing from this frame.")
    ap.add_argument("--start_oid", type=int, default=None, help="Start tracing from this object_id.")

    ap.add_argument("--mode", choices=["last_frame_all", "single"], default="last_frame_all",
                    help="last_frame_all: trace all objects in the last frame; single: trace one target.")
    ap.add_argument("--max_skip", type=int, default=3, help="Max frames to skip when searching parents (default=3).")
    ap.add_argument("--max_steps", type=int, default=100000, help="Safety cap for trace length.")
    ap.add_argument("--emit_merge_excel", action="store_true",
                    help="Also emit merge-branches excel (areas of all parent branches before merges).")

    ap.add_argument("--recursive", action="store_true", help="If --input is a folder, recursively scan subfolders.")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    excel_files = iter_excel_files(args.input, recursive=args.recursive)
    if not excel_files:
        raise RuntimeError(f"No excel files found at: {args.input}")

    is_batch = os.path.isdir(args.input)

    ok, fail = 0, 0
    failed_list = []

    for f in excel_files:
        try:
            if is_batch:
                rel = os.path.relpath(f, args.input)
                subname = sanitize_relpath_as_folder(rel)  # 保留相对层级信息
                out_dir_for_this = os.path.join(args.outdir, subname)
            else:
                base = os.path.splitext(os.path.basename(f))[0]
                out_dir_for_this = os.path.join(args.outdir, base)

            outputs = process_one_excel(args, input_file=f, out_dir_for_this_file=out_dir_for_this)

            ok += 1
            print(f"[OK] {os.path.basename(f)} -> {out_dir_for_this}")
            for k, v in outputs.items():
                print(f"      - {k}: {v}")
        except Exception as e:
            fail += 1
            failed_list.append((f, str(e)))
            print(f"[FAIL] {f}: {e}")

    print("\n========== SUMMARY ==========")
    print(f"Input: {args.input}")
    print(f"Outdir: {args.outdir}")
    print(f"Total excels: {len(excel_files)}")
    print(f"OK: {ok}, FAIL: {fail}")
    if failed_list:
        print("Failed files:")
        for f, msg in failed_list:
            print(f"  - {f}: {msg}")


if __name__ == "__main__":
    main()
