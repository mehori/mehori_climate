Pythonを使って、気候学・気象学で利用する図を量産するためのパッケージ。

```
from mehori_climate import LatlonPlot, FigurePanel
import matplotlib.pyplot as plt
import xarray as xr

# open data
ds = xr.open_dataset('../../data/slp.djf.nc', decode_times=False)

# 0. read data
slp = ds["psl"]
my_data = slp[0,:,:]

# 1. Create the figure panel
fig   = FigurePanel(figsize=(16, 6))　← ここで、図をいれるパネルをつくる
plot1 = LatlonPlot(fig)　　　　　　　　　← パネルには複数図を入れられる。

# 2. Chain your methods　　　　　　　　　← 図自体はこのように下から上に要素を追加する
(plot1
    .add_shade(my_data, vnum=17, color="RdBu_r", withLine=True, linecolor="white")
    .add_coastlines(fillland=False)　　← 適宜、引数でカスタマイズする
    .add_gridlines()
    .add_colorbar()
    .add_title(title="Difference Map")
    .add_corner(left_text="Scenario A", right_text="[DJF]")
)

# 3. Render plot
fig.render(ofile="output.test01.png")　← 最後にレンダリングをする。ファイルがなければ表示
```

