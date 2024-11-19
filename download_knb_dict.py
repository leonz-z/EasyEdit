# 验证SDK token
from modelscope.hub.api import HubApi

api = HubApi()
api.login("508fd238-de98-4395-946f-9c1489c5a066")
# 数据集下载
from modelscope.msdatasets import MsDataset

ds = MsDataset.load(
    "CaiJichang/knb-dict-bs", split="train"
)  # 您可按需配置 subset_name、split，参照“快速使用”示例代码
