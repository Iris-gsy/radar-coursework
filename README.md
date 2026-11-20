# 雷达目标探测与回波仿真系统

电子信息系统综合实验。项目包含 LFM 基础测距、晴雨雾阵列仿真、KuRALS-PD 实测车辆检测以及 HarmonyOS 看板。

## 项目内容

| 目录 | 内容 |
|---|---|
| `matlab/` | 基础测距和 8 通道阵列仿真 |
| `real_data/` | KuRALS-PD 数据预处理、车辆检测和数据划分清单 |
| `dashboard/` | HarmonyOS 工程、Python HTTP 中继与示例帧 |
| `reference/` | 仿真与实测参考结果 |
| `report/` | 最终实验报告 |

## MATLAB 仿真

在项目根目录打开 MATLAB。基础测距使用 Signal Processing Toolbox 中的 `findpeaks`。

```matlab
addpath('matlab'); simulation
addpath('matlab'); setenv('RADAR_DASHBOARD_UPLOAD','0'); rador
```

阵列脚本输出 300 帧，结果写入 `outputs/array_sim/`。参考批次检出 434 个目标，产生 112 个虚警和 61 个漏检，检测概率为 87.7%。复现配置使用参考批次中的 22 dB 旁瓣抑制峰值差。

## KuRALS-PD 实测车辆检测

使用 Python 3.13。通过 [KuRALS 官方仓库](https://github.com/lihua199710/KuRALS#prepare-the-kurals-dataset) 的数据入口获取 KuRALS-PD，解压清单中的两个 LandCar 序列。`--input` 指向包含这两个序列的目录。每条序列包含 `range_doppler_numpy/` 和 `annotations/dense/`。预处理脚本按清单读取固定的 500 帧训练、200 帧验证和 500 帧独立测试数据。

```powershell
python -m pip install -r .\real_data\requirements.txt
python .\real_data\prepare_landcar.py --input "D:\Datasets\KuPulseV3" --output .\real_data\KuRALS_real_prepared
python .\real_data\kurals_vehicle_detector.py --data .\real_data\KuRALS_real_prepared --out .\outputs\kurals_vehicle
```

已有本项目的 `KuRALS_real_prepared` 数据时，可直接用其路径作为 `--data`。

| 独立测试方案 | 检出 | 检测率 | F1 | 平均每帧虚警 |
|---|---:|---:|---:|---:|
| 单候选 | 266 | 53.2% | 0.5320 | 0.468 |
| 多候选 | 328 | 65.6% | 0.2817 | 3.002 |

## HarmonyOS 看板

使用 DevEco Studio 打开 `dashboard/HarmonyOS`。在项目根目录启动中继服务，再用另一终端发送示例帧。模拟器访问宿主机的地址在入口页 `Index.ets` 中设置。

```powershell
python .\dashboard\radar_server.py
# 在另一终端运行
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/frame' -Method Post -ContentType 'application/json' -InFile '.\dashboard\sample_frame.json'
```

在 MATLAB 中将 `RADAR_DASHBOARD_UPLOAD` 设为 `1` 后运行 `rador`，可逐帧上传仿真结果。
