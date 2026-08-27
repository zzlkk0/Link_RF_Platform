# 独立验收套件

该目录只包含验收代码，不修改 rf_ranging_model 主模型。

运行：

    cd I:\FYP_RF_Platform
    python -B validation\acceptance_suite.py --output outputs\acceptance

验收结果：

- outputs\acceptance\ACCEPTANCE_REPORT.md
- outputs\acceptance\acceptance_results.json
- outputs\acceptance\simulation_monte_carlo.csv

状态定义：

- PASS：当前证据满足明确门槛。
- PARTIAL：只覆盖一部分要求，不能宣称完整合规。
- FAIL：已执行验收，但结果超过门槛。
- NOT TESTED：缺少实现、标准测试向量、硬件或实测数据。
