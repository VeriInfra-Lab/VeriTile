import bench.tritonbench_g.uniform_sampling.Proofs.StoreExecution
import bench.tritonbench_g.uniform_sampling.Proofs.TupleExecution

/-! All four actual source stores, in source order, connected to the
independent memory specification. Stores preserve the registers needed by
subsequent streams; no stream or inactive-lane mask is omitted.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.StorePhase
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def wordName (f : Fin 4) : RegName := TupleExecution.fieldName (Formula.field f.val)

def offsetName (f : Fin 4) : RegName :=
  match f.val with | 0 => "off_0" | 1 => "off_1" | 2 => "off_2" | _ => "off_3"

def body (y : RegionName) (N B : Nat) : List Stmt := [
  Algorithm.storeStmt y N B "r0" "off_0", Algorithm.storeStmt y N B "r1" "off_1",
  Algorithm.storeStmt y N B "r2" "off_2", Algorithm.storeStmt y N B "r3" "off_3"]

theorem source_slice (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) :
    (Algorithm.kernel swapped y N seed offset lower upper B).body.drop 19 = body y N B := rfl

theorem field_regs {B : Nat} (hB : 0 < B) (pid : Nat) (base : BlockState)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : ℝ) (f : Fin 4) :
    (Memory.storeField pid base y N B seed offset lower upper f).regs = base.regs := by
  have h := (Memory.writeSequence_state y N B seed offset lower upper
    (Memory.fieldAddresses pid B f) base).1
  rw [Memory.writeSequence_field hB] at h
  exact h

private theorem four_steps {a b c d : Stmt} {s0 s1 s2 s3 s4 : BlockState}
    (ha : stepStmt a s0 = some s1) (hb : stepStmt b s1 = some s2)
    (hc : stepStmt c s2 = some s3) (hd : stepStmt d s3 = some s4) :
    stepStmts [a, b, c, d] s0 = some s4 := by
  simp only [stepStmts, ha, hb, hc, hd]

theorem stores_exec {B : Nat} (hB : 0 < B) (pid : Nat) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : ℝ) (s : BlockState)
    (hv : ∀ f : Fin 4, s.regs .real [B] (wordName f) = some ⟨fun i =>
      some (ScalarSpec.value seed offset (SourceDomain.counterIndex pid B i.1.val)
        (Formula.field f.val) lower upper)⟩)
    (ho : ∀ f : Fin 4, s.regs .nat [B] (offsetName f) =
      some ⟨fun i => SourceDomain.outputIndex pid B f i.1⟩) :
    stepStmts (body y N B) s = some (Memory.storeBlock pid s y N B seed offset lower upper) := by
  let t0 := Memory.storeField pid s y N B seed offset lower upper 0
  let t1 := Memory.storeField pid t0 y N B seed offset lower upper 1
  let t2 := Memory.storeField pid t1 y N B seed offset lower upper 2
  have h0 := StoreExecution.store_field pid y N seed offset lower upper 0
    (wordName 0) (offsetName 0) s (hv 0) (ho 0)
  have h1 := StoreExecution.store_field pid y N seed offset lower upper 1
    (wordName 1) (offsetName 1) t0
    (by simpa only [t0, field_regs hB] using hv 1)
    (by simpa only [t0, field_regs hB] using ho 1)
  have h2 := StoreExecution.store_field pid y N seed offset lower upper 2
    (wordName 2) (offsetName 2) t1
    (by simpa only [t1, t0, field_regs hB] using hv 2)
    (by simpa only [t1, t0, field_regs hB] using ho 2)
  have h3 := StoreExecution.store_field pid y N seed offset lower upper 3
    (wordName 3) (offsetName 3) t2
    (by simpa only [t2, t1, t0, field_regs hB] using hv 3)
    (by simpa only [t2, t1, t0, field_regs hB] using ho 3)
  rw [Memory.storeBlock_scatter hB]
  exact four_steps h0 h1 h2 h3

#axiomsClean source_slice
#axiomsClean field_regs
#axiomsClean stores_exec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.StorePhase
