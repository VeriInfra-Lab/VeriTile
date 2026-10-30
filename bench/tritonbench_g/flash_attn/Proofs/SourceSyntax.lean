import bench.tritonbench_g.flash_attn.Kernels

/-! Compatibility source decomposition. Authoritative ordinary triton programs
are in the selected Kernels entry; these parts only support execution proofs. -/
namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile.Triton
noncomputable section

def parts (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel p).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 64,[],[]⟩

theorem source_parts (p : Parameters) :
    (kernel p).surfaceBody=(parts p).before ++
      [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
        (.constNat 64) (parts p).body] ++ (parts p).after := rfl

#axiomsClean source_parts
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
