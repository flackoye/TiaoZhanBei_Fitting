"""统一算法比较脚本。"""
import logging, time
from pathlib import Path
import pandas as pd, yaml
from run_fitting import process

COMBINATIONS=[
 (1,"Hampel + PCHIP","hampel","pchip"),
 (2,"Hampel + Savitzky-Golay","hampel","savgol"),
 (3,"Median + PCHIP","median","pchip"),
 (4,"LOWESS","hampel","lowess"),
 (5,"CubicSpline","hampel","cubicspline"),
]
def main():
 logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
 with open("config.yaml",encoding="utf-8") as f: cfg=yaml.safe_load(f)
 rows=[]
 for algorithm_id,algorithm_name,detector,method in COMBINATIONS:
  suffix=f"_{algorithm_id:02d}_{detector}_{method}"
  _,s=process(cfg,method,detector,limit_one=False,make_plots=False,output_suffix=suffix,generate_dense=False)
  s.insert(0,"algorithm_id",algorithm_id)
  s.insert(1,"algorithm_name",algorithm_name)
  rows.append(s)
 out=pd.concat(rows,ignore_index=True).sort_values(["algorithm_id","model","source_file","experiment_id","target"])
 out.to_csv(Path(cfg["output_dir"])/"method_comparison.csv",index=False,encoding="utf-8-sig")
 summary=(out.groupby(["algorithm_id","algorithm_name","target"],as_index=False)
  .agg(group_count=("model","size"),outlier_count=("outlier_count","sum"),
       raw_max_jump_mean=("raw_max_adjacent_jump","mean"),fitted_max_jump_mean=("fitted_max_adjacent_jump","mean"),
       raw_roughness_mean=("raw_roughness","mean"),fitted_roughness_mean=("fitted_roughness","mean"),
       high_confidence_mae=("high_confidence_mae","mean"),peak_retention=("peak_retention","mean"),
       overshoot_count=("overshoot","sum"),runtime_seconds=("runtime_seconds","max")))
 summary.to_csv(Path(cfg["output_dir"])/"algorithm_comparison_summary.csv",index=False,encoding="utf-8-sig")
if __name__=="__main__": main()
