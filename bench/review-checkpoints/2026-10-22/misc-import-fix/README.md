# 两题 Misc 联合导入修复（2026-10-22）

`chunk_bwd_dqkg` 与 `matmul_kernel` 的主入口和全部题级 `Misc/` 入口现在可以一起导入。保留当前主定理和 Tensor/native 入口名称，将旧 host 结果放在独立 namespace。

| 文件 | 新名称（省略共同项目 namespace） |
| --- | --- |
| [chunk_bwd_dqkg/Misc/HostReader.lean](../../../repairs/tritonbench_g/chunk_bwd_dqkg/Misc/HostReader.lean) | `ChunkBwdDqkg.HostReader.realApplication/nativeApplication` |
| [matmul_kernel/Misc/Legacy/HostCorrect.lean](../../../repairs/tritonbench_g/matmul_kernel/Misc/Legacy/HostCorrect.lean) | `MatmulKernel.Legacy.HostCorrect.correct` |
| [matmul_kernel/Misc/Legacy/HostFPEquiv.lean](../../../repairs/tritonbench_g/matmul_kernel/Misc/Legacy/HostFPEquiv.lean) | `MatmulKernel.Legacy.HostFPEquiv.fp_equiv` |

原 `ChunkBwdDqkg.Reader.*`、`MatmulKernel.correct` 与 `MatmulKernel.fp_equiv` 保留。没有指向旧 host 名称的仓库 Lean 调用需要迁移；同名现代接口不能再兼任旧 host 别名。

最初的 matmul 联合检查首先在 `correct` 处停止；修复后暴露出 legacy host 与当前 native 的 `fp_equiv` 重名，本次一并修复。两次失败证据均保留，没有把后一个冲突隐藏成检查通过。

验证包括三个改动模块的针对性构建、两个原先失败的完整联合导入，以及显式检查新旧接口可同时引用。chunk 联合文件还保留原来的两个完整 IO `rfl` 对照；matmul 联合文件包含全部六个 `Misc/` 文件（含 `CandidateCorrect`）。两组最终检查均 exit 0。

逐文件对照确认：三个源码文件只改 namespace 和对应的 `end`，声明及证明正文未改，checked host API 保留。选定 kernel、Contract、IO、主 Correct/FPEquiv 的源哈希与唯一逐题清单保持原样，数值规则、延期状态和完成统计不变。已更新题级 README，关闭相应 [TODO](../../../TODO.md)；历史 IO 审查与失败日志保持可追溯。

来源、哈希和检查日志见 [audit.json](audit.json)。这是两题的命名与导入修复，不改变此前单列的全库共享代兼容状态，也未进行 GPU 实测或独立全源码回放。
