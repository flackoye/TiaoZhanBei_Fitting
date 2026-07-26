"""局部离群点检测与可审计修正。"""
from __future__ import annotations
import numpy as np
import pandas as pd

def _rolling(y, window):
    s = pd.Series(y)
    med = s.rolling(window, center=True, min_periods=max(3, window//2)).median().to_numpy()
    mad = pd.Series(np.abs(y-med)).rolling(window, center=True, min_periods=max(3, window//2)).median().to_numpy()
    return med, mad

def detect(y, method="hampel", window=9, mad_threshold=3.5, iqr_multiplier=1.5, jump_threshold=None):
    y = np.asarray(y, float); med, mad = _rolling(y, window)
    if method == "hampel":
        scale = 1.4826 * mad
        deviation=np.abs(y-med)
        # 离散等级常使局部 MAD=0；此时仅把偏离局部多数值的候选点交给连续段保护逻辑。
        mask = np.where(scale > 1e-12, deviation > mad_threshold*scale, deviation > 1e-12)
        reason = "hampel_mad"
    elif method == "median":
        mask = np.abs(y-med) > mad_threshold * np.nanmedian(np.abs(y-med))
        reason = "median_filter"
    elif method == "iqr":
        s=pd.Series(y); q1=s.rolling(window,center=True,min_periods=3).quantile(.25); q3=s.rolling(window,center=True,min_periods=3).quantile(.75)
        mask=((s<q1-iqr_multiplier*(q3-q1))|(s>q3+iqr_multiplier*(q3-q1))).to_numpy(); reason="local_iqr"
    elif method == "jump":
        d=np.abs(np.diff(y,prepend=y[0])); t=jump_threshold or (np.nanmedian(d)+mad_threshold*1.4826*np.nanmedian(np.abs(d-np.nanmedian(d))))
        mask=(d>t)&(np.r_[d[1:],0]>t); reason="first_difference_jump"
    else: raise ValueError(f"未知离群检测方法: {method}")
    # 连续异常更可能是真实边界；只修孤立点，避免抹掉连续突变区。
    # 整段连续候选（>=3）视作真实边界/平台，不修正；只处理短促孤立脉冲。
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
    for a,b in zip(starts,ends):
        if b-a+1 >= 3: mask[a:b+1]=False
    return mask.astype(bool), np.where(mask, reason, ""), med

def correct(y, mask, method="interpolate", confidence=None, local_median=None):
    y=np.asarray(y,float); out=y.copy(); idx=np.arange(len(y)); valid=~mask & np.isfinite(y)
    if not mask.any(): return out
    if method == "median": out[mask]=np.asarray(local_median)[mask]
    elif method == "interpolate":
        out[mask]=np.interp(idx[mask],idx[valid],y[valid]) if valid.any() else y[mask]
    elif method == "confidence_weight":
        conf=np.nan_to_num(np.asarray(confidence,float),nan=.5); target=np.asarray(local_median)
        out[mask]=conf[mask]*y[mask]+(1-conf[mask])*target[mask]
    else: raise ValueError(f"未知修正方法: {method}")
    return out
