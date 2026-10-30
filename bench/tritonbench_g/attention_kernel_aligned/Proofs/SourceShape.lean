import bench.tritonbench_g.attention_kernel_aligned.Proofs.Parameters
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prologue p ++
      [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
        (.constNat p.blockN) (body p)] ++ epilogue p := rfl

theorem full_source_lengths (p : Parameters) :
    (prologue p).length=23 ∧ (body p).length=16 ∧ (epilogue p).length=3 := ⟨rfl,rfl,rfl⟩
#axiomsClean source_decomposition
#axiomsClean full_source_lengths
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Source

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prologue p ++
      [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
        (.constNat p.blockN) (body p)] ++ epilogue p := rfl

theorem full_source_lengths (p : Parameters) :
    (prologue p).length=23 ∧ (body p).length=16 ∧ (epilogue p).length=3 := ⟨rfl,rfl,rfl⟩
#axiomsClean source_decomposition
#axiomsClean full_source_lengths
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
