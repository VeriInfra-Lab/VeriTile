import bench.tritonbench_g.rms_matmul_rbe.Proofs.RawInput

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem liveKey_reference {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (k : Fin 64) :
    liveKey p (valuesAt a fp8 p s t) t k ↔ t*64+k.val<p.K := by
  unfold liveKey
  rw [valuesAt_keys]
  exact Nat.lt_sub_iff_add_lt'

theorem xTile_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (k : Fin 64) :
    xTile a p s (valuesAt a fp8 p s t) t (row,k,PUnit.unit)=
      RmsMatmulRbe.masked a.toScalar.toOps (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)).x t k := by
  simp [xTile,liveKey_reference,valuesAt_xptrs,RmsMatmulRbe.masked,
    parameters,rawRow,z.literal]

theorem rmsTile_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t batch row : Nat) (k : Fin 64) :
    rmsTile a fp8 p s (valuesAt a fp8 p s t) t k=
      RmsMatmulRbe.masked a.toScalar.toOps (parameters p epsilon theta)
        (rawRow a fp8 p s batch row).rms t k := by
  cases fp8 <;> by_cases hk : t*64+k.val<p.K
  all_goals
    simp [rmsTile,liveKey_reference,valuesAt_rmsptrs,RmsMatmulRbe.masked,
      parameters,rawRow,hk,z.half,z.rmsE5]

theorem weightTile_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t batch row : Nat) (k n : Fin 64) :
    weightTile a fp8 p s (valuesAt a fp8 p s t) t (k,n,PUnit.unit)=
      RmsMatmulRbe.masked a.toScalar.toOps (parameters p epsilon theta)
        ((rawRow a fp8 p s batch row).weight ((programCol p s*64+n.val)%p.N)) t k := by
  cases fp8 <;> by_cases hk : t*64+k.val<p.K
  all_goals
    simp [weightTile,liveKey_reference,valuesAt_wptrs,RmsMatmulRbe.masked,
      parameters,rawRow,hk,z.half,z.weightE5]

theorem scaledTile_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (t : Nat) (row : Fin 16) (k : Fin 64) :
    scaledTile a fp8 p s (valuesAt a fp8 p s t) t (row,k,PUnit.unit)=
      RmsMatmulRbe.scaled a.toScalar.toOps (parameters p epsilon theta)
        (rawRow a fp8 p s (s.pids 0) ((programRow p s*16+row.val)%p.M)) t k := by
  simp only [scaledTile,RmsMatmulRbe.scaled,xTile_reference a z fp8 p s epsilon theta,
    rmsTile_reference a z fp8 p s epsilon theta t (s.pids 0)
      ((programRow p s*16+row.val)%p.M)]

#axiomsClean liveKey_reference
#axiomsClean xTile_reference
#axiomsClean rmsTile_reference
#axiomsClean weightTile_reference
#axiomsClean scaledTile_reference
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
