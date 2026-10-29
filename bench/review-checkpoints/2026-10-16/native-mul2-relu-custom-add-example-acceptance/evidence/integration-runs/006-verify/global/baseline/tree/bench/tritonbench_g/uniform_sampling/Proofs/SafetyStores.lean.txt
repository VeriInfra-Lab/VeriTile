import bench.tritonbench_g.uniform_sampling.Proofs.StorePhase
import bench.tritonbench_g.uniform_sampling.Proofs.FlatMemory

/-! The exact masked addresses of all four actual stores lie inside their
finite allocation. Local register values and offsets determine execution;
no complete-kernel success or trace-safety premise is assumed.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.SafetyStores
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

theorem store_safe {B : Nat} (bounds : RegionBounds) (pid : Nat) (y : RegionName)
    (N : Nat) (f : Fin 4) (wordName offsetName : RegName) (s : BlockState)
    (ho : s.regs .nat [B] offsetName = some ⟨fun i => SourceDomain.outputIndex pid B f i.1⟩)
    (hy : ∀ j : Fin B, SourceDomain.outputIndex pid B f j < N →
      SourceDomain.outputIndex pid B f j < bounds y) :
    Stmt.TraceSafe bounds (Algorithm.storeStmt y N B wordName offsetName) s := by
  simp [-Tile.scalar_eta, Algorithm.storeStmt, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    MemAccess.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
    evalOp.eq_def, ho, Tile.cop, Tile.scalar, ComparableDType.lt, Region.cast]
  exact hy

private theorem four_safe (bounds : RegionBounds) {a b c d : Stmt} {s0 s1 s2 s3 : BlockState}
    (ha : stepStmt a s0 = some s1) (hb : stepStmt b s1 = some s2)
    (hc : stepStmt c s2 = some s3)
    (sa : Stmt.TraceSafe bounds a s0) (sb : Stmt.TraceSafe bounds b s1)
    (sc : Stmt.TraceSafe bounds c s2) (sd : Stmt.TraceSafe bounds d s3) :
    Stmt.TraceSafeList bounds [a, b, c, d] s0 := by
  cases hd : stepStmt d s3 <;> simp [Stmt.TraceSafeList, ha, hb, hc, hd, sa, sb, sc, sd]

theorem stores_safe {B : Nat} (hB : 0 < B) (bounds : RegionBounds) (pid : Nat)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : ℝ) (s : BlockState)
    (hv : ∀ f : Fin 4, s.regs .real [B] (StorePhase.wordName f) = some ⟨fun i =>
      some (ScalarSpec.value seed offset (SourceDomain.counterIndex pid B i.1.val)
        (Formula.field f.val) lower upper)⟩)
    (ho : ∀ f : Fin 4, s.regs .nat [B] (StorePhase.offsetName f) =
      some ⟨fun i => SourceDomain.outputIndex pid B f i.1⟩)
    (hy : FlatMemory.Bounds y N B pid bounds) :
    Stmt.TraceSafeList bounds (StorePhase.body y N B) s := by
  let t0 := Memory.storeField pid s y N B seed offset lower upper 0
  let t1 := Memory.storeField pid t0 y N B seed offset lower upper 1
  let t2 := Memory.storeField pid t1 y N B seed offset lower upper 2
  have hr0 : t0.regs = s.regs := StorePhase.field_regs hB pid s y N seed offset lower upper 0
  have hr1 : t1.regs = s.regs :=
    (StorePhase.field_regs hB pid t0 y N seed offset lower upper 1).trans hr0
  have hr2 : t2.regs = s.regs :=
    (StorePhase.field_regs hB pid t1 y N seed offset lower upper 2).trans hr1
  have hv1 := hv 1
  have hv2 := hv 2
  have ho1 := ho 1
  have ho2 := ho 2
  have ho3 := ho 3
  rw [← hr0] at hv1 ho1
  rw [← hr1] at hv2 ho2
  rw [← hr2] at ho3
  have h0 := StoreExecution.store_field pid y N seed offset lower upper 0
    (StorePhase.wordName 0) (StorePhase.offsetName 0) s (hv 0) (ho 0)
  have h1 := StoreExecution.store_field pid y N seed offset lower upper 1
    (StorePhase.wordName 1) (StorePhase.offsetName 1) t0 hv1 ho1
  have h2 := StoreExecution.store_field pid y N seed offset lower upper 2
    (StorePhase.wordName 2) (StorePhase.offsetName 2) t1 hv2 ho2
  exact four_safe bounds h0 h1 h2
    (store_safe bounds pid y N 0 _ _ s (ho 0) (hy.output 0))
    (store_safe bounds pid y N 1 _ _ t0 ho1 (hy.output 1))
    (store_safe bounds pid y N 2 _ _ t1 ho2 (hy.output 2))
    (store_safe bounds pid y N 3 _ _ t2 ho3 (hy.output 3))

#axiomsClean store_safe
#axiomsClean stores_safe

end
end VeriTile.Bench.TritonBenchG.UniformSampling.SafetyStores
