"""独立运行入口：默认 Hampel + PCHIP。"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
from data_loader import load_predictions, collapse_duplicate_depths
from outlier_detection import detect, correct
from fitting_methods import fit_curve, constrain, dense_interpolate
from evaluation import metrics
from visualization import plot_group, safe_name

LOG=logging.getLogger("fitting")
def process(config, method=None, detector=None, limit_one=False, make_plots=True, output_suffix="", generate_dense=True):
    start=time.perf_counter(); out_cfg=config["outlier"]; fit_cfg=config["fitting"]
    method=method or fit_cfg["method"]; detector=detector or out_cfg["method"]
    df,depth,groups=load_predictions(config["input_file"],config)
    output=Path(config["output_dir"]); output.mkdir(parents=True,exist_ok=True)
    result=[]; summaries=[]; dense_results=[]
    iterator=df.groupby(groups,sort=False,dropna=False)
    for gi,(key,g0) in enumerate(iterator):
        if limit_one and gi>0: break
        g,ndup=collapse_duplicate_depths(g0,depth)
        if ndup: LOG.warning("分组 %s 聚合 %d 条重复深度记录",key,ndup)
        x=g[depth].to_numpy(float)
        # 检测是否已加载不确定性权重
        use_weights="damage_weight" in df.columns and "stress_weight" in df.columns
        fitted_map={}; cleaned_map={}; raw_map={}; masks={}; reasons=[]
        x_dense=dd=ss=None
        for kind,col in [("damage","pred_damage_level"),("stress","pred_stress_mpa")]:
            raw=g[col].to_numpy(float); jump=out_cfg.get(f"jump_threshold_{kind}")
            mask,reason,med=detect(raw,detector,out_cfg["window_size"],out_cfg["mad_threshold"],out_cfg.get("iqr_multiplier",1.5),jump)
            if use_weights:
                wcol=f"{kind}_weight"
                conf=g[wcol].to_numpy(float) if wcol in g.columns else g["state_confidence"].to_numpy(float)
            else:
                conf=g["state_confidence"].to_numpy(float)
            cleaned=correct(raw,mask,out_cfg["correction"],conf,med)
            raw_map[kind]=raw; cleaned_map[kind]=cleaned
            fitted=constrain(fit_curve(x,cleaned,method,fit_cfg,conf,continuous_blend=use_weights),raw,kind,fit_cfg)
            fitted_map[kind]=fitted; masks[kind]=mask; reasons.append(reason)
            met=metrics(raw,fitted,mask,conf,config["evaluation"]["high_confidence_threshold"],config["evaluation"]["peak_quantile"])
            met.update(dict(zip(groups,key if isinstance(key,tuple) else (key,)))); met["target"]=kind; met["method"]=f"{detector}+{method}"; summaries.append(met)
        g["raw_damage"]=g["pred_damage_level"]; g["corrected_damage"]=cleaned_map["damage"]; g["fitted_damage"]=fitted_map["damage"]
        g["raw_stress"]=g["pred_stress_mpa"]; g["corrected_stress"]=cleaned_map["stress"]; g["fitted_stress"]=fitted_map["stress"]
        g["damage_outlier"]=masks["damage"]; g["stress_outlier"]=masks["stress"]
        g["outlier_reason"]=[";".join(filter(None,[reasons[0][i],reasons[1][i]])) for i in range(len(g))]
        g["correction_method"]=np.where(g.damage_outlier|g.stress_outlier,out_cfg["correction"],"none")
        g["fitting_method"]=f"{detector}+{method}"
        # 聚合只服务于拟合；按深度映射回所有原始行，保证输入字段和行均被保留。
        added=["corrected_damage","corrected_stress","fitted_damage","fitted_stress","damage_outlier","stress_outlier","outlier_reason","correction_method","fitting_method"]
        mapped=g[[depth]+added]
        original=g0.merge(mapped,on=depth,how="left",validate="many_to_one")
        original["raw_damage"]=original["pred_damage_level"]
        original["raw_stress"]=original["pred_stress_mpa"]
        result.append(original)
        if generate_dense:
            step=float(fit_cfg.get("output_depth_step") or 0.1)
            if step <= 0: raise ValueError("output_depth_step 必须大于 0")
            # 严格等间距；若终点不落在网格上，则保留终点之前的最后一个等距点。
            x_dense=np.arange(float(x[0]),float(x[-1])+step*1e-7,step)
            x_dense=x_dense[x_dense<=float(x[-1])+step*1e-7]
            # 最终连续曲线只采用“Savitzky-Golay结果 → PCHIP保形致密插值”。
            # 不生成未经最终平滑的独立 PCHIP/CubicSpline 曲线，避免交付口径混乱。
            dd=constrain(dense_interpolate(x,fitted_map["damage"],x_dense,"pchip"),raw_map["damage"],"damage",fit_cfg)
            ss=constrain(dense_interpolate(x,fitted_map["stress"],x_dense,"pchip"),raw_map["stress"],"stress",fit_cfg)
            block=pd.DataFrame({depth:x_dense,"continuous_damage":dd,"continuous_stress":ss,
                                "fitting_method":"hampel+savgol+pchip_dense","depth_step_cm":step})
            for col,value in zip(groups,key if isinstance(key,tuple) else (key,)):
                block[col]=value
            dense_results.append(block[groups+[depth,"continuous_damage","continuous_stress","fitting_method","depth_step_cm"]])
        if make_plots:
            sfx=output_suffix or ("_sample" if limit_one else "")
            folder=output/"plots"/safe_name(key[0])/safe_name(key[-1])/(safe_name(key[1])+sfx)
            title=" | ".join(map(str,key))
            plot_group(x,g.raw_damage.to_numpy(),g.fitted_damage.to_numpy(),g.damage_outlier.to_numpy(),"损伤等级",folder/"damage_raw_vs_fitted.png",title,config["plot"],x_dense,dd)
            plot_group(x,g.raw_stress.to_numpy(),g.fitted_stress.to_numpy(),g.stress_outlier.to_numpy(),"应力 (MPa)",folder/"stress_raw_vs_fitted.png",title,config["plot"],x_dense,ss)
            both=masks["damage"]|masks["stress"]
            plot_group(x,g.raw_damage.to_numpy(),g.fitted_damage.to_numpy(),both,"损伤/离群位置",folder/"outlier_locations.png",title,config["plot"],x_dense,dd)
    elapsed=time.perf_counter()-start
    res=pd.concat(result,ignore_index=True); summary=pd.DataFrame(summaries); summary["runtime_seconds"]=elapsed
    suffix=output_suffix or ("_sample" if limit_one else "")
    res.to_csv(output/f"fitted_predictions{suffix}.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(output/f"fitting_summary{suffix}.csv",index=False,encoding="utf-8-sig")
    if generate_dense and dense_results:
        pd.concat(dense_results,ignore_index=True).to_csv(output/f"fitted_curve_dense{suffix}.csv",index=False,encoding="utf-8-sig")
    LOG.info("完成 %d 行、%d 个汇总项，用时 %.2f 秒",len(res),len(summary),elapsed)
    return res,summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--one-group",action="store_true"); ap.add_argument("--no-plots",action="store_true"); ap.add_argument("--suffix",default="",help="输出文件后缀，用于区分不同权重模式"); args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler("fitting.log",encoding="utf-8"),logging.StreamHandler()])
    with open(args.config,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    process(cfg,limit_one=args.one_group,make_plots=not args.no_plots,output_suffix=args.suffix)
if __name__=="__main__": main()
