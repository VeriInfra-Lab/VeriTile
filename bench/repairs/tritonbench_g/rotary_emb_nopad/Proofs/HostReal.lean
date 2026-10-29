import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Host

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
/-- Whole guarded source wrapper over arbitrary independent real values. -/
theorem rotary_emb_nopad_correct (c : Checked.Call) (metadata : Physical.Bytes) (m : Memory ℝ)
    (accepted : Checked.check c (Physical.indices c.request metadata)=true) :
    ∃ result,Host.realInvoke c metadata m=some result ∧
      Host.RealClaim c (Physical.indices c.request metadata) m result :=
  Host.complete_real c metadata m accepted

#axiomsClean rotary_emb_nopad_correct
end VeriTile.Bench.Repairs.RotaryEmbNopad
