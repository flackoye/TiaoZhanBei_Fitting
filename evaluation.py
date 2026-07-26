"""拟合质量指标。"""
from __future__ import annotations
import numpy as np

def total_variation(y): return float(np.nansum(np.abs(np.diff(y))))
def max_jump(y): return float(np.nanmax(np.abs(np.diff(y)))) if len(y)>1 else 0.
def roughness(y): return float(np.nanmean(np.diff(y,n=2)**2)) if len(y)>2 else 0.

def metrics(raw, fitted, outliers, confidence, high_threshold=.8, peak_quantile=.95):
    raw=np.asarray(raw,float); fitted=np.asarray(fitted,float); c=np.asarray(confidence,float)
    hi=np.nan_to_num(c,nan=-1)>=high_threshold
    peak=raw>=np.nanquantile(raw,peak_quantile)
    peak_denom=float(np.nanmean(raw[peak])) if peak.any() else 0
    return {"n_points":len(raw),"outlier_count":int(np.sum(outliers)),"outlier_ratio":float(np.mean(outliers)),
      "raw_total_variation":total_variation(raw),"fitted_total_variation":total_variation(fitted),
      "raw_max_adjacent_jump":max_jump(raw),"fitted_max_adjacent_jump":max_jump(fitted),
      "raw_roughness":roughness(raw),"fitted_roughness":roughness(fitted),
      "high_confidence_mae":float(np.mean(np.abs(fitted[hi]-raw[hi]))) if hi.any() else np.nan,
      "peak_retention":float(np.nanmean(fitted[peak])/peak_denom) if peak.any() and peak_denom else 1.,
      "overshoot":bool((fitted<np.nanmin(raw)-1e-9).any() or (fitted>np.nanmax(raw)+1e-9).any())}
