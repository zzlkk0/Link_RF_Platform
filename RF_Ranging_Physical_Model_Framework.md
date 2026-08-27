# RF Ranging Physical Model Simulation Framework

## Python Main Framework + MATLAB Validation Framework

## 1. 项目目标

本项目旨在建立一个面向 **BLE Channel Sounding / UWB Ranging / RFIC Co-design** 的统一无线传播仿真框架。

核心目标是建立：

```text
Physical Environment
        ↓
Propagation Channel
        ↓
Received RF Signal
        ↓
RFIC Non-Idealities
        ↓
Ranging Algorithm
        ↓
Distance Error
```

最终用于研究：

- 2.4 GHz Bluetooth Channel Sounding
- UWB ToF/CIR ranging
- NLOS detection
- AI/SNN correction
- RFIC architecture optimization

---

# 2. Overall Architecture

```text
                Environment Model
                       |
                       v
              Propagation Engine
          +------------+-------------+
          |                          |
          v                          v
       h(t) CIR                  H(f) Channel
       Time Domain               Frequency Domain
          |                          |
          +------------+-------------+
                       |
                       v
              RFIC Impairment Model
       +---------------+----------------+
       |               |                |
       v               v                v
    Phase Error     Group Delay       Noise
    IQ mismatch     CFO               Phase Noise
                       |
                       v
              Ranging Processing
       +---------------+---------------+
       BLE PBR                    UWB ToF
       Phase slope                First path
       estimation                 detection
                       |
                       v
              AI / SNN Correction
                       |
                       v
                  Final Distance
```

---

# 3. Software Architecture

推荐目录结构：

```text
RF_Ranging_Model/
│
├── channel/
│   ├── free_space.py
│   ├── two_ray.py
│   ├── multipath.py
│   ├── stochastic_channel.py
│   └── material.py
│
├── signal/
│   ├── waveform.py
│   ├── modulation.py
│   └── noise.py
│
├── rfic/
│   ├── sparameter.py
│   ├── phase_error.py
│   ├── iq_imbalance.py
│   ├── group_delay.py
│   └── pll_noise.py
│
├── ble/
│   ├── pbr.py
│   └── channel_sounding.py
│
├── uwb/
│   ├── cir.py
│   ├── tof.py
│   └── first_path.py
│
├── ai/
│   ├── nlos_classifier.py
│   ├── correction.py
│   └── snn.py
│
├── matlab_validation/
├── experiments/
└── README.md
```

---

# 4. Mathematical Foundation

## 4.1 Multipath Channel Model

无线信道：

\[
h(t)
=
\sum_{l=0}^{L-1}
\alpha_l e^{j\theta_l}
\delta(t-\tau_l)
\]

其中：

| 参数 | 含义 |
|---|---|
| \(L\) | 路径数量 |
| \(\alpha_l\) | 路径衰减 |
| \(\theta_l\) | 相位偏移 |
| \(\tau_l\) | 延迟 |

传播距离：

\[
\tau_l=\frac{d_l}{c}
\]

---

# 5. Frequency Domain Channel

对 CIR 做 Fourier Transform：

\[
H(f)
=
\sum_l
\alpha_l
 e^{-j2\pi f\tau_l}
 e^{j\theta_l}
\]

其中：

- BLE Channel Sounding 主要使用 \(H(f)\)
- UWB 主要使用 \(h(t)\)

---

# 6. Level 0: Free Space Model

## Single Path LOS

传播：

\[
H(f)
=
\frac{\lambda}{4\pi d}
 e^{-j2\pi fd/c}
\]

输出：

- amplitude
- phase
- delay

验证：

\[
d_{\mathrm{estimated}} \approx d
\]

---

# 7. Level 1: Two-Ray Model

加入一次反射：

```text
        reflector
           *
          /
TX ---------------- RX
```

信号：

\[
H(f)=H_{\mathrm{LOS}}(f)+H_{\mathrm{reflection}}(f)
\]

反射路径：

\[
d_r>d_{\mathrm{LOS}}
\]

产生：

- phase distortion
- amplitude fading

用于：

- multipath analysis
- BLE phase error

---

# 8. Level 2: Deterministic Multipath

建立：

```text
TX
 |
 +------ direct path
 |
 +------ wall reflection
 |
 +------ floor reflection
 |
 +------ object reflection
 |
RX
```

模型：

\[
H(f)
=
\sum_{i=0}^{N}
\alpha_i
 e^{-j2\pi f\tau_i}
\]

输出：

- CFR
- phase residual
- delay spread

---

# 9. Level 3: Statistical Channel Model

采用：

## Saleh-Valenzuela Model

用于：

- UWB
- Indoor channel

形式：

\[
h(t)
=
\sum_l
\sum_k
 a_{k,l}
 e^{j\phi_{k,l}}
\delta(t-T_l-\tau_{k,l})
\]

参数：

- cluster arrival
- ray arrival
- delay spread
- fading

参考：

- IEEE 802.15.4a UWB Channel Model

---

# 10. RFIC Impairment Model

传播信道：

\[
H_{\mathrm{channel}}(f)
\]

实际接收：

\[
H_{\mathrm{total}}(f)
=
H_{\mathrm{channel}}(f)
H_{\mathrm{RFIC}}(f)
\]

---

# 11. IQ Imbalance

理想：

\[
I+jQ
\]

实际会出现：

- I/Q gain mismatch
- I/Q phase mismatch
- image rejection degradation

重点观察：

\[
\Delta G_{IQ}
\]

和：

\[
\Delta \phi_{IQ}
\]

---

# 12. Group Delay Model

RFIC 相位：

\[
\phi_{\mathrm{RF}}(f)
\]

Group delay：

\[
\tau_g
=
-\frac{1}{2\pi}
\frac{d\phi}{df}
\]

重点指标：

\[
\Delta\tau_g
=
\tau_{\max}-\tau_{\min}
\]

影响：

- UWB pulse distortion
- BLE phase-ranging bias

---

# 13. Phase Noise / CFO

实际测量：

\[
\phi_{\mathrm{meas}}
=
\phi_{\mathrm{channel}}
+
\phi_{\mathrm{RF}}
+
\phi_{\mathrm{PLL}}
+
\phi_{\mathrm{CFO}}
+
n
\]

其中：

| 项目 | 来源 |
|---|---|
| channel phase | 环境 |
| RF phase | receiver |
| PLL phase | oscillator |
| CFO | frequency mismatch |
| \(n\) | noise |

---

# 14. BLE Channel Sounding Model

## Phase-Based Ranging

频率：

\[
f_1,f_2,\ldots,f_N
\]

测量：

\[
\phi(f)
\]

理想：

\[
\phi(f)
=
-2\pi f\tau+\phi_0
\]

估计：

\[
\tau
=
-\frac{1}{2\pi}
\frac{d\phi}{df}
\]

距离：

\[
d=c\tau
\]

---

# 15. UWB ToF Model

接收：

\[
y(t)
=
x(t)*h(t)+n(t)
\]

目标：

寻找：

\[
\tau_{\mathrm{first}}
\]

距离：

\[
d=c\tau_{\mathrm{first}}
\]

重点：

- first-path detection
- CIR processing
- NLOS correction

---

# 16. AI / SNN Interface

输入可包括：

```text
Phase residual
RSSI
SNR
CFR amplitude
CFR phase
Delay spread
CFO
IQ mismatch
```

输出：

```text
NLOS probability
Distance correction Δd
Confidence
```

最终：

\[
d=d_0+\Delta d
\]

---

# 17. MATLAB Validation

MATLAB 用于：

## 1. 标准模型验证

例如：

- IEEE 802.15.4a
- UWB channel

## 2. Ray Tracing

输入：

- 3D environment
- material
- antenna

输出：

- path loss
- delay
- phase

## 3. RF Toolbox

验证：

- S-parameters
- group delay
- phase response

---

# 18. RFIC Co-Design Flow

```text
Cadence Virtuoso
      |
      v
LNA / Mixer / BBF / PLL
      |
      v
S21(f), NF, phase, group delay
      |
      v
Python
      |
      v
H_RFIC(f)
      |
      v
Propagation Channel Model
      |
      v
Ranging Algorithm
      |
      v
Ranging Accuracy / Energy Trade-off
```

---

# 19. Experiment Plan

## Experiment 1: LOS Accuracy

变量：

- distance
- SNR

观察：

- phase error
- ToF error
- distance error

## Experiment 2: Multipath

变量：

- reflection coefficient
- path delay
- number of paths

观察：

- CFR distortion
- CIR distortion
- ranging bias

## Experiment 3: NLOS

障碍：

- wall
- human
- furniture
- metal

观察：

- attenuation
- excess delay
- phase residual
- first-path degradation

## Experiment 4: RFIC Impact

改变：

- NF
- phase error
- group delay ripple
- I/Q mismatch
- PLL phase noise
- CFO

观察：

\[
RF\ metric
\rightarrow
Ranging\ error
\]

---

# 20. Final Research Question

核心问题：

> How can RF front-end architecture and lightweight intelligence be co-designed to achieve low-power high-accuracy ranging under realistic multipath environments?

进一步可拆分为：

1. RF front-end 的 phase/group-delay nonideality 会造成多少 ranging error？
2. 哪些 RF 指标真正主导 BLE PBR 和 UWB ToF 的精度？
3. 是否可以降低 Analog/RF 功耗，再通过 digital/AI correction 补偿？
4. SNN 是否能利用稀疏的 channel events 实现更低能耗的 NLOS detection / ranging correction？

---

# 21. Development Roadmap

```text
Version 1
Free-space physical model
      ↓
Version 2
Two-ray / deterministic multipath
      ↓
Version 3
Statistical channel model
      ↓
Version 4
BLE PBR + UWB ToF
      ↓
Version 5
RFIC nonidealities
      ↓
Version 6
AI correction
      ↓
Version 7
SNN / event-driven accelerator
      ↓
Version 8
RTL / ASIC implementation
```

---

# 22. Recommended First Implementation

第一版建议只完成四个核心模块：

```text
free_space.py
      ↓
two_ray.py
      ↓
multipath.py
      ↓
H(f) / h(t) generator
```

完成后必须验证：

### Test 1

单径 LOS 下：

\[
d_{\mathrm{estimated}}\approx d_{\mathrm{true}}
\]

### Test 2

Two-ray 下：

- CFR 出现 amplitude fading
- phase 不再严格线性

### Test 3

Multi-path 下：

- CIR 出现多个 peaks
- BLE PBR 出现 ranging bias
- UWB first-path detection 难度增加

---

# 23. Python 与 MATLAB 分工

## Python：主框架

负责：

- propagation engine
- deterministic multipath
- statistical channel
- H(f)
- CIR
- BLE PBR
- UWB ToF
- RFIC impairment import
- Monte Carlo
- AI / SNN
- data visualization

## MATLAB：验证框架

负责：

- 标准 channel model 对照
- Communications Toolbox
- RF Toolbox
- Ray Tracing
- 标准 UWB / propagation 模型交叉验证

总体策略：

> **Python 做主模型，MATLAB 做标准模型和工具箱验证。**

---

# 24. Final Goal

最终目标不是简单模拟无线传播，而是建立：

\[
\boxed{
Environment
\rightarrow
Channel
\rightarrow
RFIC
\rightarrow
Ranging
\rightarrow
AI/SNN
\rightarrow
Distance
}
\]

的统一跨层仿真平台。

该平台同时支持：

- BLE Channel Sounding
- UWB ranging
- RFIC optimization
- NLOS analysis
- AI/SNN correction
- Analog–Digital co-design

并作为后续 FYP、论文和芯片设计的基础。
