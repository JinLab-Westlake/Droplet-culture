#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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
        # excel里可能是 5.0
        return int(float(str(x).strip()))
    except Exception:
        return default


def parse_parent_ids(val) -> List[int]:
    """
    parent_ids 可能是：
      - FALSE / True / 空
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
    # 允许逗号/空格/分号分隔
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


# -------------------------
# Core tracing
# -------------------------
@dataclass
class TraceRow:
    frame_idx: int
    object_id: int
    mask_area: float
    timestamp: Optional[str] = None


def build_frame_index(
    df: pd.DataFrame,
    frame_col: str,
    obj_col: str,
) -> Dict[int, Dict[int, List[int]]]:
    """
    返回：frame_idx -> object_id -> 行索引list（同一帧同一object_id可能多行）
    """
    idx: Dict[int, Dict[int, List[int]]] = {}
    for i, r in df.iterrows():
        f = safe_int(r.get(frame_col), None)
        o = safe_int(r.get(obj_col), None)
        if f is None or o is None:
            continue
        idx.setdefault(f, {}).setdefault(o, []).append(i)
    return idx


def pick_best_row(
    df: pd.DataFrame,
    row_indices: List[int],
    area_col: str,
) -> int:
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
) -> Optional[Tuple[int, int]]:
    """
    在 cur_frame-1 ... cur_frame-max_skip 里找 parent。
    返回 (parent_frame, parent_row_index)，规则：
      - 从近到远扫描帧
      - 在某一帧内，如果 parent_ids 多个，选 mask_area 最大的那个 parent 行
      - 找到就立刻返回
    """
    for pf in range(cur_frame - 1, cur_frame - max_skip - 1, -1):
        if pf not in frame_map:
            continue
        best_row = None
        best_area = -np.inf

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

        if best_row is not None:
            return pf, best_row

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
) -> List[TraceRow]:
    """
    从 (start_frame, start_oid) 开始向前追溯。
    """
    # 起点行：同一帧同一oid多行，取面积最大
    rows = frame_map.get(start_frame, {}).get(start_oid, [])
    if not rows:
        raise ValueError(f"start (frame={start_frame}, oid={start_oid}) not found in input.")
    cur_idx = pick_best_row(df, rows, area_col=area_col)

    trace: List[TraceRow] = []
    steps = 0
    while cur_idx is not None and steps < max_steps:
        steps += 1
        r = df.loc[cur_idx]
        cur_frame = safe_int(r.get(frame_col), None)
        cur_oid = safe_int(r.get(obj_col), None)
        if cur_frame is None or cur_oid is None:
            break

        trace.append(
            TraceRow(
                frame_idx=cur_frame,
                object_id=cur_oid,
                mask_area=safe_float(r.get(area_col), np.nan),
                timestamp=(str(r.get(ts_col)) if ts_col and not pd.isna(r.get(ts_col)) else None),
            )
        )

        parents = parse_parent_ids(r.get(parent_col))
        if not parents:
            break

        found = find_parent_row_with_skip(
            df=df,
            frame_map=frame_map,
            cur_frame=cur_frame,
            parent_ids=parents,
            area_col=area_col,
            max_skip=max_skip,
        )
        if found is None:
            break

        _, parent_row = found
        cur_idx = parent_row

    # 当前 trace 是 latest->earliest（因为一直往前找）
    # 反转成 earliest->latest 更便于分析面积变化
    trace = list(reversed(trace))
    return trace


# -------------------------
# Export
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
                }
            )
    return pd.DataFrame(rows)


def long_df_to_matrix_area(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    你要的格式：frame_idx，idx(1-n)列的数值为area
    """
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
    # 关键点：Excel 不要用 read_csv；read_excel 不涉及 utf-8 解码问题
    df = pd.read_excel(path, sheet_name=sheet if sheet else 0, engine=None)
    # 统一列名
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Trace tracking targets backward using parent_ids with frame skipping.")
    ap.add_argument("-i", "--input", required=True, help="Input tracking Excel file (.xlsx).")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory.")
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
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    df = load_tracking_excel(args.input, sheet=args.sheet)

    # 基础检查
    for c in [args.frame_col, args.obj_col, args.area_col, args.parent_col]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}. Existing columns: {list(df.columns)}")

    ts_col = args.ts_col if args.ts_col in df.columns else None

    frame_map = build_frame_index(df, frame_col=args.frame_col, obj_col=args.obj_col)

    traces: List[List[TraceRow]] = []

    if args.mode == "single":
        if args.start_frame is None or args.start_oid is None:
            raise RuntimeError("--mode single requires --start_frame and --start_oid")
        tr = trace_back_one(
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

    else:
        # last_frame_all：找最大 frame_idx 的那一帧，追这帧出现的所有 object_id
        all_frames = [f for f in frame_map.keys()]
        if not all_frames:
            raise RuntimeError("No valid rows with frame/object parsed.")
        last_frame = int(max(all_frames))
        oids = sorted(frame_map[last_frame].keys())
        for oid in oids:
            try:
                tr = trace_back_one(
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
            except Exception:
                # 某些目标可能断得很早或数据不全，允许跳过
                continue

    # 导出
    df_long = traces_to_long_df(traces)
    out_long = os.path.join(args.outdir, "traces_long.xlsx")
    df_long.to_excel(out_long, index=False)

    df_mat = long_df_to_matrix_area(df_long)
    out_mat = os.path.join(args.outdir, "traces_area_matrix.xlsx")
    df_mat.to_excel(out_mat, index=False)

    print(f"[OK] traced {len(traces)} targets")
    print(f"[OK] wrote: {out_long}")
    print(f"[OK] wrote: {out_mat}")


if __name__ == "__main__":
    main()
