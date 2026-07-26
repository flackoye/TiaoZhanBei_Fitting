"""拟合前后和离群点图。"""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _font in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if Path(_font).exists():
        font_manager.fontManager.addfont(_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

def safe_name(v): return re.sub(r'[^\w.-]+','_',str(v),flags=re.UNICODE)[:100]
def plot_group(x, raw, fitted, outliers, ylabel, output, title, cfg, x_dense=None, continuous=None):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(12,5)); step=max(1,len(x)//cfg.get("max_points",4000)); sl=slice(None,None,step)
    plt.plot(x[sl],raw[sl],color="0.55",lw=.8,alpha=.75,label="原始预测")
    plt.plot(x[sl],fitted[sl],color="#42a5f5",lw=1.0,alpha=.85,label="Savitzky-Golay平滑")
    if x_dense is not None and continuous is not None:
        plt.plot(x_dense,continuous,color="#0d47a1",lw=1.5,label="PCHIP连续曲线")
    m=np.asarray(outliers,bool); plt.scatter(np.asarray(x)[m],np.asarray(raw)[m],c="#d32f2f",s=18,zorder=3,label="离群点")
    plt.xlabel("累计孔深 (cm)"); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.2); plt.legend(); plt.tight_layout(); plt.savefig(output,dpi=cfg.get("dpi",140)); plt.close()
