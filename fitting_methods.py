"""连续拟合方法及置信度自适应融合。"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

def _odd_window(n, requested, poly=2):
    w=min(int(requested), n if n%2 else n-1); w=max(poly+2+(poly+2)%2, w); return w if w<=n else (n if n%2 else n-1)

def fit_curve(x, y, method, cfg, confidence=None):
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(y)
    if n < 3: return y.copy()
    if method == "pchip": fitted=PchipInterpolator(x,y,extrapolate=False)(x)
    elif method == "savgol":
        w=_odd_window(n,cfg.get("savgol_window_length",9),cfg.get("savgol_polyorder",2)); fitted=savgol_filter(y,w,cfg.get("savgol_polyorder",2),mode="interp") if w>=3 else y.copy()
    elif method == "lowess": fitted=lowess(y,x,frac=cfg.get("lowess_frac",.08),it=2,return_sorted=False)
    elif method == "cubicspline": fitted=CubicSpline(x,y,bc_type="natural")(x)
    else: raise ValueError(f"未知拟合方法: {method}")
    conf=np.nan_to_num(np.asarray(confidence if confidence is not None else np.full(n,.5),float),nan=.5)
    threshold=cfg.get("confidence_threshold",.7)
    blend=np.where(conf>=threshold,cfg.get("high_confidence_blend",.2),cfg.get("low_confidence_blend",.8))
    return (1-blend)*y+blend*fitted

def constrain(fitted, raw, kind, cfg):
    raw=np.asarray(raw,float); lo,hi=np.nanmin(raw),np.nanmax(raw); margin=(hi-lo)*cfg.get("range_margin_ratio",.05)
    if kind=="damage": return np.clip(fitted,lo,hi)
    if cfg.get("stress_nonnegative",True): lo=max(0,lo)
    return np.clip(fitted,lo,hi+margin)

def dense_interpolate(x, y, x_dense, method):
    """在新的等间距深度网格上计算连续插值，不与原始采样点混合。"""
    x=np.asarray(x,float); y=np.asarray(y,float); x_dense=np.asarray(x_dense,float)
    if len(x) < 2:
        return np.full_like(x_dense,y[0] if len(y) else np.nan,dtype=float)
    if method == "pchip":
        return PchipInterpolator(x,y,extrapolate=False)(x_dense)
    if method == "cubicspline":
        return CubicSpline(x,y,bc_type="natural",extrapolate=False)(x_dense)
    raise ValueError(f"致密网格仅支持 PCHIP/CubicSpline，收到: {method}")
