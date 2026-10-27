from models.dl.mlp_columns import LAYOUTS
from models.dl.mlp_resnet import ResNetModel

# 逐马二分类五塔残差 MLP。四个任务各训一份；含赔率 132 列，无赔率 131 列。


class Model(ResNetModel):
    layout = LAYOUTS[5]
    log_name = 'tower: race + horse + jockey + trainer + pair'
