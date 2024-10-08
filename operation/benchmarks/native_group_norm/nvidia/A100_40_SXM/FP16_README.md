# 参评AI芯片信息

* 厂商：Nvidia

* 产品名称：A100
* 产品型号：A100-40GiB-SXM
* TDP：400W

# 所用服务器配置

* 服务器数量：1
* 单服务器内使用卡数: 1
* 服务器型号：DGX A100
* 操作系统版本：Ubuntu 20.04.4 LTS
* 操作系统内核：linux5.4.0-113
* CPU：AMD EPYC7742-64core
* docker版本：20.10.16
* 内存：1TiB
* 服务器间AI芯片直连规格及带宽：此评测项不涉及服务期间AI芯片直连

# 算子库版本

https://github.com/FlagOpen/FlagGems. Commit ID: 3c10679326b32ea5f037db50cc397d41c0ff1934

# 评测结果

## 核心评测结果

| 评测项  | correctness | TFLOPS(cpu wall clock) | TFLOPS(kernel clock) | FU(FLOPS Utilization)-cputime | FU-kerneltime |
| ---- | -------------- | -------------- | ------------ | ------ | ----- |
| flaggems | True    | 0.31TFLOPS       | 0.3TFLOPS        | 0.1% | 0.1% |
| nativetorch | True    | 0.41TFLOPS      | 0.37TFLOPS      | 0.13%      | 0.12%    |

## 其他评测结果

| 评测项  | cputime | kerneltime | cputime吞吐 | kerneltime吞吐 | 无预热时延 | 预热后时延 |
| ---- | -------------- | -------------- | ------------ | ------------ | -------------- | -------------- |
| flaggems | 229.3us       | 236.54us        | 4361.11op/s | 4227.54op/s | 4851723.17us | 361.73us |
| nativetorch | 173.42us       | 189.44us        | 5766.26op/s | 5278.72op/s | 16604.37us | 239.69us |

## 能耗监控结果

| 监控项  | 系统平均功耗  | 系统最大功耗  | 系统功耗标准差 | 单机TDP | 单卡平均功耗 | 单卡最大功耗 | 单卡功耗标准差 | 单卡TDP |
| ---- | ------- | ------- | ------- | ----- | ------------ | ------------ | ------------- | ----- |
| nativetorch监控结果 | 1404.0W | 1404.0W | 0.0W   | /     | 161.0W       | 170.0W      | 9.0W        | 400W  |
| flaggems监控结果 | 1404.0W | 1404.0W | 0.0W   | /     | 148.0W       | 165.0W      | 23.34W        | 400W  |

## 其他重要监控结果

| 监控项  | 系统平均CPU占用 | 系统平均内存占用 | 单卡平均温度 | 单卡最大显存占用 |
| ---- | --------- | -------- | ------------ | -------------- |
| nativetorch监控结果 | 0.547%    | 1.22%   | 35.07°C       | 2.622%        |
| flaggems监控结果 | 0.576%    | 1.227%   | 34.12°C       | 3.797%        |
