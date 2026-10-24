import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Host

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
/-- The only conditional numerical boundary is the local FP32 mul/neg/fma
realization. Complete source execution, ownership, objects and bytes are proved. -/
theorem rotary_emb_nopad_fp32 (actual target : Native.Instructions)
    (ports : Native.Obligations actual target) (c : Checked.Call) (m : Physical.Bytes)
    (accepted : Checked.check c (Physical.indices c.request m)=true) :
    ∃ result,Host.invoke actual c m=some result ∧ Host.NativeClaim actual target c m result := by
  rw [←VeriTile.Bench.Repairs.RotaryEmb.Native.instructions_eq actual target ports]
  exact Host.complete_native actual c m accepted

#axiomsClean rotary_emb_nopad_fp32
end VeriTile.Bench.Repairs.RotaryEmbNopad
