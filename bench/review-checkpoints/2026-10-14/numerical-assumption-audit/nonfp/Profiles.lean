import bench.tritonbench_g.bgmv_expand_slice.Proofs.NativeFP
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BgmvExpandSlice
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
#eval (do
  let inputs : List (String × TypedSource.InputStorage) := [("fp16", .fp16), ("bf16", .bf16), ("fp32", .fp32)]
  let weights : List (String × TypedSource.WeightStorage) := [("fp16", .fp16), ("bf16", .bf16)]
  for (iname, input) in inputs do
    for (wname, weight) in weights do
      for cast in [Bool.false, Bool.true] do
        for output in [FloatDType.real, .fp32, .fp16, .bf16, .f8e4, .f8e5] do
          let k := TypedSource.original input weight cast output "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.true Bool.true
          IO.println s!"{iname}|{wname}|{cast}|{repr output}|{repr ((NativeFP.requirementsFor k).map NativeFP.Site.profile)}"
  : IO Unit)
