import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPMemory
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.GridRun
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α : Type} [Inhabited α]

abbrev Address := RegionName × Nat
abbrev Heap (α : Type) := Address → Cell α
def entry (heap : Heap α) (pid count : Nat) : State α :=
  ⟨fun r o => heap (r,o),fun _ _ _ => none,fun _ => pid,fun _ => count,fun d _ _ => defaultValue d⟩
abbrev Output (h : Call) := Fin h.cols ⊕ Unit
def address (h : Call) (row : Fin h.rows) : Output h → Address
  | .inl j => ("y",row.val*h.cols+j.val)
  | .inr _ => ("r",row.val)
def writes (h : Call) (row : Fin h.rows) (a : Address) : Prop :=
  (∃ j : Fin h.cols,a=(("y":RegionName),row.val*h.cols+j.val)) ∨ a=(("r":RegionName),row.val)
def reads (_h : Call) (_row : Fin _h.rows) (a : Address) : Prop := a.1="x" ∨ a.1="w"
def expected (M : Algebra α) (h : Call) (heap : Heap α) (row : Fin h.rows) : Output h → Cell α
  | .inl j => FPReference.output M h
      (FPMemory.raw h (entry heap row.val h.rows) "x" h.cols)
      (FPMemory.raw h (entry heap row.val h.rows) "w" 0) ⟨j.val,j.isLt.trans_le h.covered⟩
  | .inr _ => FPReference.savedInv M h (FPMemory.raw h (entry heap row.val h.rows) "x" h.cols)
def execute (M : Algebra α) (h : Call) (row : Fin h.rows) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (engine M) (Kernels.forward h) (entry heap row.val h.rows)).map
    (fun v a => v.mem a.1 a.2)

theorem inside (h : Call) (row : Fin h.rows) (i : Output h) : writes h row (address h row i) := by
  cases i with
  | inl j => exact Or.inl ⟨j,rfl⟩
  | inr _ => exact Or.inr rfl

theorem separate (h : Call) (a b : Fin h.rows) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  constructor
  · intro hb
    rcases ha with ⟨i,ha⟩|ha <;> rcases hb with ⟨j,hb⟩|hb
    · have eq := congrArg Prod.snd (ha.symm.trans hb)
      exact ne (Fin.ext (Geometry.row_injective h a.val b.val i j eq).1)
    · have eq := congrArg Prod.fst (ha.symm.trans hb)
      exact (by decide : ("y":RegionName)≠"r") eq
    · have eq := congrArg Prod.fst (ha.symm.trans hb)
      exact (by decide : ("r":RegionName)≠"y") eq
    · exact ne (Fin.ext (congrArg Prod.snd (ha.symm.trans hb)))
  · intro hr
    rcases ha with ⟨j,rfl⟩|rfl <;> simp [reads] at hr

theorem step_complete (M : Algebra α) (h : Call) (row : Fin h.rows) (heap : Heap α) :
    ∃ final,execute M h row heap=some final ∧
      (∀ i,final (address h row i)=expected M h heap row i) ∧
      (∀ addr,¬writes h row addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hr,hf⟩ := FPMemory.forward_complete M h (entry heap row.val h.rows)
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],?_,?_⟩
  · intro i
    cases i with
    | inl j => exact hy j
    | inr _ => exact hr
  · rintro ⟨q,o⟩ outside
    apply hf
    rintro (⟨same,j,eq⟩|⟨same,eq⟩)
    · exact outside (Or.inl ⟨j,Prod.ext same eq⟩)
    · exact outside (Or.inr (Prod.ext same eq))

theorem expected_stable (M : Algebra α) (h : Call) (row : Fin h.rows) (a b : Heap α)
    (stable : ∀ addr,reads h row addr → a addr=b addr) (i : Output h) :
    expected M h a row i=expected M h b row i := by
  have x : FPMemory.raw h (entry a row.val h.rows) "x" h.cols=
      FPMemory.raw h (entry b row.val h.rows) "x" h.cols := by
    funext j
    exact congrArg (Cell.read .real) (stable ("x",row.val*h.cols+j.val) (Or.inl rfl))
  have w : FPMemory.raw h (entry a row.val h.rows) "w" 0=
      FPMemory.raw h (entry b row.val h.rows) "w" 0 := by
    funext j
    exact congrArg (Cell.read .real) (stable ("w",row.val*0+j.val) (Or.inr rfl))
  cases i <;> simp only [expected,x,w]

def run (M : Algebra α) (h : Call) := GridRun.run (execute M h) (List.finRange h.rows)

theorem complete (M : Algebra α) (h : Call) (heap : Heap α) :
    ∃ final,run M h heap=some final ∧
      (∀ row : Fin h.rows,∀ i,final (address h row i)=expected M h heap row i) ∧
      (∀ addr,(∀ row : Fin h.rows,¬writes h row addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute M h) (fun _ => Output h) (address h) id
    (expected M h) (reads h) (writes h) (inside h) (List.finRange h.rows) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete M h w)
    (fun w _ => expected_stable M h w) heap
  exact ⟨final,he,fun row => ho row (List.mem_finRange row),fun addr out => hf addr (fun row _ => out row)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridForward
