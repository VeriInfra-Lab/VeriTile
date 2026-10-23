import bench.repairs.tritonbench_g.layer_norm_triton.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Source execution uses ordinary total scalar operations and an explicitly
selected local typed reduction. The adapter supplies only the tl.sum primitive,
never a kernel result or successful program execution. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceModel
open VeriTile.Triton FP.Structural
noncomputable section

structure Model (α : Type) where
  scalar : Algebra α
  sumTyped : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (ax : Fin shape.length) → (keepDims : Bool) → Values α dtype.toTileDType shape →
    Values α dtype.toTileDType (TileShape.reduceShape shape ax keepDims)

def engine (m : Model α) : Algebra α :=
  {m.scalar with reduceSumTyped := fun p d {_} a k v => some (m.sumTyped p d a k v)}

theorem range_one [Inhabited α] (m : Algebra α) (idx : RegName)
    (N B : Nat) (positive : 0<N) (covers : N≤B) (body : List ComputeStmt) (s : State α) :
    FP.Structural.range m idx 0 N B body s=
      run m body (s.setReg idx .nat [] (fun _ => 0)) := by
  have hb : B≠0 := by omega
  rw [FP.Structural.range]
  simp only [hb,if_false,positive,if_true]
  cases h : run m body (s.setReg idx .nat [] (fun _ => 0)) with
  | none => rfl
  | some t =>
    simp only [bind,Option.bind,Nat.zero_add]
    change FP.Structural.range m idx B N B body t=some t
    rw [FP.Structural.range]
    simp [hb,show ¬B<N by omega]

#axiomsClean range_one
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceModel
