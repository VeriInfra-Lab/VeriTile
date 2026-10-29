# RBE 原核实验交接 v2

根代理复核了 **45 项工具测试、16 个新离线编译配置**，独立生成的 13 个参考文件逐字相同；实际 GPU 为 **NOT_RUN，0 次 launch**。本包不新增 Lean 定理、公开规格或完成任务数。

初版独立审查发现两处验收器缺口，本版已修复并加入拒错测试：预选架构不能在结果中消失，离线或未执行结果不能伪标为 GPU 环境已绑定。原核、用例、全部内存足迹和 7 项数值义务保持不变；初版及审查记录留存。

原 pow、sin、cos 都由 helper 内实际执行。当前编译图包含 libdevice pow/trig、近似/FTZ 原语和融合末端运算，不能由理想 Real 函数代替。旧 Nat token 与真实 i32 还存在明确回绕差异：2147483647+1 的 signed i32 结果为 −2147483648，不能把无限 Nat 对应当作待实验确认的普遍等式。

[7 项具体绑定缺口](candidate/numeric-obligations.json)、[编译图记录](candidate/compiler-observations.json)、[详细运行说明](candidate/README.zh-CN.md)、[根验收](root-review.json)。

在 SM80/SM90 GPU 上，从本目录执行：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/src/run.py run --reference candidate/final-validation/reference --output /tmp/rbe-native-capture
python3 -B candidate/src/verify_capture.py /tmp/rbe-native-capture
```

输出目录须不存在。工具采集 10 个完整用例 × 两种融合配置，包含全部分配的前后 raw words、双 mask、非单位步长、origin=8、原地别名和全部 guard/frame。当前数值比较明确为 NOT_ESTABLISHED；成功采集仍只是 CHARACTERIZATION。
