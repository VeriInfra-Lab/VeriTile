import bench.tritonbench_g.layer_norm_triton.Proofs.PythonSource
import bench.tritonbench_g.layer_norm_triton.Proofs.Support.KernelEntryAdapters
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

theorem masked_setReg {ι : Type} (r : RegionName) (offset : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) (n : RegName) (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    l.foldl (fun t i => if active i then t.write r (offset i) (value i) else t) (s.setReg n d sh v) =
      (l.foldl (fun t i => if active i then t.write r (offset i) (value i) else t) s).setReg n d sh v := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split
    · simpa only [show (s.setReg n d sh v).write r (offset i) (value i) =
        (s.write r (offset i) (value i)).setReg n d sh v from rfl] using ih (s.write r (offset i) (value i))
    · exact ih s

@[simp] theorem nat_product_address (a b : Nat) : ((a : Int)*b).toNat=a*b := by
  rw [← Nat.cast_mul]
  rfl
@[simp] theorem nat_sum_address (a b : Nat) : ((a : Int)+b).toNat=a+b := by
  rw [← Nat.cast_add]
  rfl
@[simp] theorem nat_product_sum_address (a b c : Nat) : ((a : Int)*b+c).toNat=a*b+c := by
  rw [← Nat.cast_mul,← Nat.cast_add]
  rfl

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

theorem signed_range_nat_step [Inhabited α] (m : Algebra α) (idx : RegName)
    (cur stop stride : Nat) (body : List ComputeStmt) (s : State α) :
    FP.Structural.rangeInt m idx (cur : Int) (stop : Int) stride body s =
      if stride=0 then some s else if cur<stop then do
        let t ← run m body (s.setReg idx .int [] (fun _ => (cur : Int)))
        FP.Structural.rangeInt m idx ((cur+stride : Nat) : Int) (stop : Int) stride body t
      else some s := by
  rw [FP.Structural.rangeInt]
  simp only [Int.ofNat_lt,Nat.cast_add]

theorem signed_range_one [Inhabited α] (m : Algebra α) (idx : RegName)
    (N B : Nat) (positive : 0<N) (covers : N≤B) (body : List ComputeStmt) (s : State α) :
    FP.Structural.rangeInt m idx 0 (N : Int) B body s=
      run m body (s.setReg idx .int [] (fun _ => 0)) := by
  have hb : B≠0 := by omega
  rw [FP.Structural.rangeInt]
  simp only [hb,if_false,Int.natCast_pos.mpr positive,if_true]
  cases h : run m body (s.setReg idx .int [] (fun _ => 0)) with
  | none => rfl
  | some t =>
    simp only [bind,Option.bind,Int.zero_add]
    rw [FP.Structural.rangeInt]
    simp [hb,show ¬(B : Int)<N by exact_mod_cast (Nat.not_lt.mpr covers)]

#axiomsClean signed_range_one
#axiomsClean range_one
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceModel
