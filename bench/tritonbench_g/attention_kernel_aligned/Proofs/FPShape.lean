import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
open VeriTile.Triton

/-- The initial query is genuinely read from fp16 storage, with the original
unchecked block-pointer load and unspecified padding flag. The empty checked
axes list means the padding branch is never observed. -/
theorem query_storage (p : Parameters) :
    ((prologue p).drop 11 |>.take 1) =
      [.assign .fp16 [p.blockM,p.head] "q" (.alg
        (.load .fp16 (.blockPtr (.ref .blockPtr [p.blockM,p.head] "Q_block_ptr") [])
          .none .unspecified))] := rfl

theorem score_dot_source (p : Parameters) :
    ((body p).drop 3 |>.take 1) =
      [.assign .fp16 [p.blockM,p.blockN] "qk" (.alg
        (.add .fp16 Broadcast.nil.consSame.consSame (.ref .fp16 [p.blockM,p.blockN] "qk")
          (.dotTyped .fp16 .fp16 .fp16 (batch:=[]) (M:=p.blockM) (K:=p.head) (N:=p.blockN) (.ref .fp16 [p.blockM,p.head] "q")
            (.ref .fp16 [p.head,p.blockN] "k") {outDType:=some .fp16})))] := rfl

/-- Legacy mathematical hypotheses are not imported by the new source. The
actual integer loop ranges over every block, with no per-step premise. -/
theorem accepted_range (p : Parameters) :
    ((source p).surfaceBody.drop 19 |>.take 1) =
      [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
        (.constNat p.blockN) (body p)] := rfl

#axiomsClean query_storage
#axiomsClean score_dot_source
#axiomsClean accepted_range
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
