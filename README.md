# RF Ranging 跨层物理仿真平台

本目录已经把 RF_Ranging_Physical_Model_Framework.md 中的架构落成一个可运行、可测试、可扩展的平台：

    Environment / Paths
              |
              v
    Channel H(f) and CIR
              |
              v
    RFIC non-idealities / S2P
              |
        +-----+-----+
        |           |
        v           v
      BLE PBR     UWB ToF
        |           |
        +-----+-----+
              |
              v
       AI / SNN interface

当前版本为可复现实验基线，重点完成 Markdown 路线图的 Version 1–5，并为 Version 6–7 提供稳定接口。AI/SNN 模块可以执行，但目前是透明的启发式/轻量基线，不应当被当作已经由实测数据训练出的模型。

当前已经按以下顺序完成软件层修复并接入统一流水线：

1. BLE 模糊度感知解析、质量门控和明确拒绝；
2. BLE 基于 Matrix Pencil、BIC 阶数选择和最早显著分量的复多径解析；
3. UWB CFAR 与 `PASS / LOW_CONFIDENCE / NO_PATH`；
4. UWB 公共相位对齐的多帧 CFR 合并，以及可显式启用的保守强晚径 CLEAN 早径重检；
5. 频率相关 RFIC 复数校准 Profile；
6. 本机图形化实验与验收界面。

## 快速开始

在 PowerShell 中进入本目录：

    cd I:\FYP_RF_Platform

启动图形化平台：

    python -B -m rf_ranging_model.webapp --open

默认地址为 `http://127.0.0.1:8765/`。页面完全在本机运行，不加载外部 JavaScript/CSS；可切换 LOS、多径、低 SNR、UWB 相位对齐帧数、无校准、标量校准和频率 Profile，且同时显示原始数值、数值候选、可用结果、拒绝原因、BLE 解析模式/阶数、UWB 帧相干度和校准 Profile ID。页面还会显示仿真路径真值、BLE 多径估计和最终选中路径。

页面内置六个固定 seed 教学案例：理想 LOS、BLE 复杂多径、UWB 单帧晚径反例、UWB 四帧恢复、低帧相干度和错误校准保护。点击案例即可加载并运行真实模型；“预期观察”和“实际观察”会并列说明该案例的物理含义。距离对比图将真值、旧单径/原始候选、鲁棒候选和最终可用结果放在同一坐标轴上，多径结构图则对照真实路径与 BLE 估计路径。路径真值只来自合成仿真，用于教学和回归，真实部署中不可直接获得。

完整、桌面和手机宽度截图分别保存于 `outputs\web_visual_qa\full.png`、`outputs\web_visual_qa\desktop.png` 和 `outputs\web_visual_qa\mobile.png`。

命令行参考实验：

    python -m rf_ranging_model --output outputs\demo

运行后会生成：

- outputs\demo\demo_summary.json：距离、误差、相位残差与 NLOS 概率
- outputs\demo\paths.csv：本次实验的路径真值
- outputs\demo\ble_cfr_phase.png：BLE CFR 幅相与相位拟合
- outputs\demo\uwb_cir.png：UWB CIR 和首径检测

执行全部 Python 自检：

    python -m unittest discover -s tests -v

执行独立软件验收（数值门槛失败时退出码为 1）：

    python validation\acceptance_suite.py --output outputs\acceptance

报告会分别给出 `software_gate_status` 和 `standards_status`。前者通过只表示
本仓库列出的数值/回归门槛通过；协议标准和实测覆盖不全时，总体仍为 `PARTIAL`。

执行 MATLAB 交叉验证（不需要额外 Toolbox）：

    matlab -batch "cd('I:\FYP_RF_Platform\matlab_validation'); run_all_validations"

MATLAB 结果写入 matlab_validation\output\matlab_validation_summary.json。

## 已实现模块

| 层级 | 模块 | 当前能力 |
|---|---|---|
| Channel | channel.free_space | Friis 幅度、传播相位、时延 |
| Channel | channel.two_ray | 几何直达径与单次反射、衰落与非线性相位 |
| Channel | channel.multipath | 任意确定性路径的 H(f)、采样 CIR、RMS delay spread |
| Channel | channel.stochastic_channel | 可复现的 Saleh–Valenzuela 风格簇/射线模型 |
| Signal | signal.waveform / noise | Gaussian monocycle、复数 AWGN |
| RFIC | rfic.impairments | phase、group delay/ripple、幅度波纹、IQ、CFO、phase noise、SNR |
| RFIC | rfic.sparameter | Touchstone v1 S2P 的 S21 读取与插值 |
| RFIC | rfic.calibration | 不可变复数校准 Profile、log-magnitude/解缠相位插值、参考面/来源/ID、越界/空洞/深零点/相位混叠/过大逆增益守卫 |
| BLE | ble.channel_sounding / pbr | 官方 72 频点、全局相干模糊度搜索、残差 WLS、无模糊范围守卫 |
| BLE | ble.robust | 实际 tone-map GCD；Matrix Pencil 多径模型；BIC 阶数选择；最早显著路径；重构残差、路径竞争和挑战模型门控；`available/rejected/invalid_input` |
| UWB | uwb.cir | 单帧 CFR→CIR；多帧公共相位对齐与相干合并；帧间 coherence 证据 |
| UWB | uwb.first_path / tof | 稳健噪声估计、整窗 CFAR、首主瓣插值、可显式启用的保守 CLEAN 早径重检、`PASS/LOW_CONFIDENCE/NO_PATH` |
| UI | webapp / web_static | 本地 HTTP API、响应式界面、严格 JSON、状态与验收新鲜度显示 |
| AI | ai.features / classifier / correction | 仿真接口占位；当前含路径和注入参数真值，不计入独立精度验收 |
| SNN | ai.snn | 事件驱动 LIF 神经元基线 |
| Integration | pipeline.RangingPipeline | 同时保留 channel、RFIC、CIR、估计值和特征证据 |

## 目录结构

    I:\FYP_RF_Platform
    |-- RF_Ranging_Physical_Model_Framework.md
    |-- README.md
    |-- pyproject.toml
    |-- rf_ranging_model
    |   |-- channel
    |   |-- signal
    |   |-- rfic
    |   |-- ble
    |   |-- uwb
    |   |-- ai
    |   |-- pipeline.py
    |   |-- cli.py
    |   |-- webapp.py
    |   +-- web_static
    |-- experiments
    |-- matlab_validation
    |-- tests
    +-- outputs

## 最小 Python API

下面的例子建立 8 m LOS 加一条反射径，然后同时运行 BLE 与 UWB：

    import numpy as np
    from rf_ranging_model import C0, PropagationPath, RangingPipeline
    from rf_ranging_model.rfic import RFICImpairmentConfig

    paths = [
        PropagationPath(8.0 / C0, 1.0 + 0j, "LOS"),
        PropagationPath(8.0 / C0 + 6e-9, -0.45 + 0.1j, "wall"),
    ]
    ble_f = np.arange(2.402e9, 2.481e9, 2e6)
    uwb_f = np.linspace(3.5e9, 4.5e9, 1001)
    rfic = RFICImpairmentConfig(
        base_group_delay_s=0.2e-9,
        group_delay_ripple_s=0.08e-9,
        snr_db=30.0,
    )
    result = RangingPipeline().run(
        paths,
        ble_f,
        uwb_f,
        rfic_config=rfic,
        seed=7,
        calibrate_rfic_group_delay=True,
        rfic_group_delay_calibration_s=0.2e-9,  # 独立 VNA/环回标定值
        uwb_frame_count=4,  # 准静态弱首径场景的推荐内部基线
    )
    print(result.ble_raw_distance_m)
    print(result.ble_calibrated_distance_m)
    print(result.ble_robust.status, result.ble_robust.rejection_reason)
    print(result.uwb_raw_distance_m)
    print(result.uwb_distance_m)
    print(result.uwb_first_path.status.value)
    print(result.nlos_probability)

所有数组和参数默认使用 SI 单位：Hz、s、m、rad。通用路径的 complex_gain 不包含 exp(-j2πfτ)，这个频率相关传播相位由 channel.frequency_response 统一施加。

## 测距约定

平台的 H(f) 默认是一程传播模型，因此 BLE 和 UWB 的 path_factor 默认值都是 1：

    distance = c0 * delay / path_factor

如果输入是双向/往返测量的相位或 ToF，请显式设置 path_factor=2。这个约定避免把 BLE 协议测量过程与通用物理信道模型混在一起。

对固定 RFIC 群时延，校准后的公式为：

    distance = c0 * (detected_delay - calibration_delay) / path_factor

校准默认关闭。开启时必须显式传入 `rfic_group_delay_calibration_s`；流水线不会从
`RFICImpairmentConfig.base_group_delay_s` 读取仿真注入真值。结果同时保留 raw 与
calibrated 距离，便于审计是否重复扣除。真实硬件必须使用独立 VNA、线缆/环回或
参考面标定，不能把仿真配置值当成实测标定证据。

频率相关校准使用 `CalibrationProfile`，不能与标量校准同时开启：

    from rf_ranging_model.rfic import (
        build_calibration_profile,
        load_s2p_calibration_profile,
    )

    # measured_calibration_s21 必须来自独立扫频/环回证据，不能由本次 DUT
    # 仿真的 injected impairment 真值反算。
    ble_profile = build_calibration_profile(
        calibration_frequency_hz,
        measured_calibration_s21,
        reference_plane="antenna connector to estimator input",
        source="VNA sweep 2026-08-22 fixture A",
        maximum_expected_group_delay_s=5e-9,
    )
    result = RangingPipeline().run(
        paths,
        ble_f,
        uwb_f,
        rfic_config=rfic,
        ble_rfic_calibration_profile=ble_profile,
        uwb_rfic_calibration_profile=uwb_profile,
    )

`CalibrationResult` 会同时保存 raw、calibrated、实际应用的 correction、Profile ID、来源和参考面。Profile 禁止频段外推，也会拒绝跨越未测频率空洞、解缠条件不足、深零点或需要不稳定大增益的曲线。

BLE 的数值候选与“是否可用”是两层结果：`ble_pbr`/`ble_calibrated_distance_m` 保留候选供审计，应用层必须检查 `ble_robust.availability`。UWB 同理：`NO_PATH` 的距离为 NaN（Web JSON 表达为 `null`），`LOW_CONFIDENCE` 保留候选但不能显示为高置信 PASS。

UWB 多帧模式使用 `uwb_frame_count` 和 `uwb_frame_combining="phase_aligned"`。该模式只移除每帧的常数公共相位，不会消除频率斜率；因此它假设多帧期间信道准静态且同步/CFO 已足以保持可对齐。返回的 coherence 是质量证据，不是距离校准；低于默认门槛 0.75 时，结果不能保持高置信 `PASS`。零填充仅细化延迟网格，不提供处理增益。`RangingPipeline` 与图形平台当前都默认使用 4 帧；显式设置 `uwb_frame_count=1` 可运行单帧对照，`early_path_enhancement` 默认关闭。

BLE 全局搜索的无模糊距离由实际频道图决定：

    R_amb = c0 / (path_factor * gcd(tone_frequency_differences))

`max_distance_m` 必须严格小于 `R_amb`，否则估计器会拒绝运行。完整 1 MHz 频道格的
一程范围为 299.792458 m，往返相位范围为 149.896229 m。这里的
`path_factor=2` 测试只验证 PBR 往返相位因子，并不是 Bluetooth Mode-1 RTT
时间戳算法。

## 当前固定 seed 软件回归快照

2026-08-24 对当前代码进行独立重跑，Python 测试为 `63 / 63 PASS`；验收为 `17 PASS / 2 PARTIAL / 0 FAIL / 6 NOT TESTED`，其中 `software_gate_status=PASS`、`standards_status=PARTIAL`、`overall_status=PARTIAL`。

下面全部是固定 seed 的内部合成回归，不是真实场测：

| 场景 | 历史修复前结果 | 当前固定 seed 结果 |
|---|---|---|
| BLE 一般复多径，200 例 | 接受率 75%；接受子集 P95 约 1.623 m；18 个接受结果超过 1 m | 接受 141/200，可用率 70.5%；bias 0.0314 m、MAE 0.2233 m、RMSE 0.3000 m、P95 0.6349 m；接受结果中 `>1 m` 为 0 |
| BLE 100 m、5 dB RTT，200 例 | 旧回归为 195/200，P95 约 0.1007 m | 接受 193/200，可用率 96.5%；P95 0.1011 m；接受结果中 `>1 m` 为 0 |
| UWB 5 dB、首径/晚径幅度 0.1:1、晚径延后 10 ns | 单帧曾可能锁到约 3 m 之后的晚径并错误标记 `PASS` | 4 帧公共相位对齐：200/200 在 0.10 m 内，P95 0.0352 m，晚径锁定 0，错误晚径 `PASS` 为 0；全部保持 `LOW_CONFIDENCE` |

BLE 当前结果来自 `REG-BLE-COMPLEX-MP`，使用 Matrix Pencil 多指数模型、BIC 阶数选择、最早显著分量和模型质量拒绝。其标量校准值由独立 40 dB known-through 参考扫频估得，加入独立噪声和 20 ps 标准差的参考面偏置；pipeline 没有接收注入群时延真值。UWB 当前结果来自 `REG-UWB-WEAK-LATE-5DB`；4 帧纯噪声控制的 128 次误报为 0，中位帧 coherence 约 0.905，默认 pipeline 的 200 次复核 P95 为 0.0320 m。单帧在证据不足时仍可能保留晚径数值候选；定向压力测试显式启用保守早径增强后，错误晚径 `PASS` 为 0。该增强默认关闭，因为将单分量 CLEAN 无条件用于一般复多径会产生假早径；默认可靠路径是 4 帧方案。

这些数字证明当前给定生成器、参数范围和 seed 下的软件退化门槛通过；它们不能外推为任意房间、运动信道、未补偿 CFO、不同设备或真实 NLOS 的性能保证。

## 验证边界

自动测试覆盖以下验收条件：

1. `REG-BLE-ROBUST`：官方 72 点、缺频/notch/乱序、实际 GCD 无模糊范围、非法输入、等强双峰、纯噪声 512 次、已知 CFO 独立补偿，以及 5 dB RTT 100 m 的 200 次选择性门槛。
2. `REG-BLE-COMPLEX-MP`：固定 seed 的 200 次一般复多径回归，联合检查可用率、bias/MAE/RMSE/P95 和接受结果的 `>1 m` 比例。
3. `REG-UWB-CFAR`：0.5–149 m 独立单径、纯噪声 512 次零误报、`NO_PATH`、−12/−18 dB 转换、10 dB 弱首径/强晚径和等强近邻多径降级。
4. `REG-UWB-WEAK-LATE-5DB`：4 帧公共相位对齐合并下的 5 dB 弱首径/强晚径、晚径锁定、错误 `PASS`、帧 coherence 与纯噪声控制。
5. `NUM-RFIC-CAL / NUM-RFIC-FREQ-CAL`：标量与复数 Profile 的 `path_factor=1/2` 校准、独立异频网格、raw→calibrated 改善、六类危险 Profile 拒绝，以及过大错误标量校准的 BLE/UWB 状态拒绝。
6. `IO-WEB-UI`：真实临时 HTTP 服务的 200/400/404/413/422、严格 JSON、动态状态、raw/candidate/calibrated、NaN→null、DOM 合约、六个引导案例、距离/路径可视化数据合约与验收 `fresh/stale`；其中 UWB 单帧晚径反例、四帧恢复和低相干拒绝会由真实 API 模型重跑。
7. 原有 BLE/UWB 理想扫描、标量群时延、相位偏置、随机到达率、SNR 单调性、Touchstone v1、Monte Carlo 和 Python/MATLAB 交叉验证继续作为回归门槛。

验收 JSON 使用 schema `2.0`。只要算法、测试、界面或 README 晚于验收结果，页面会把三类结论改显示为 `STALE`，不会沿用旧 PASS。最终报告仍会把软件门槛和标准/实测覆盖分开：软件 `PASS` 不等于 Bluetooth、IEEE 或 ISO 认证。

当前版本尚未宣称完成以下工作：

- Bluetooth Channel Sounding Mode 0–3、Mode-1 RTT、包/步进时序、内部 FFO/CFO 补偿及正式测试向量的一致性验证
- IEEE 802.15.4 HRP UWB 前导码、相关器、时间戳和协议测距流程的一致性验证
- IEEE 802.15.4a 参数集的逐条标准一致性认证
- MATLAB Ray Tracing / RF Toolbox 的场景级对照
- 来自 Cadence/VNA/环回的真实 S 参数、NF、PLL 与参考平面闭环
- 带测绘真值的真实 BLE/UWB 场测和 ISO/IEC 18305 场地测试
- 用实测 NLOS 数据训练并验证 AI/SNN
- 真实能耗、RTL 或 ASIC 实现

图形界面的 NLOS 概率卡片明确标为仿真接口占位：当前特征直接使用仿真路径 delay spread 与注入 SNR/CFO/IQ 参数，因此存在 oracle 信息，不参与任何独立精度 PASS。

这些内容分别对应原始 Markdown 的后续 Version 6–8；现有接口允许后续加入，而不需要改写信道和测距主干。
