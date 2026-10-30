import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealRows
import bench.tritonbench_g.fast_rms_layernorm.Proofs.GridRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridForward
open VeriTile.Triton
noncomputable section

abbrev Address := RegionName × Nat
abbrev Heap := Address → MemCell
def read (cell : MemCell) : ℝ :=
  match cell.readAs .real with
  | some (some value) => value
  | _ => 0
def entry (heap : Heap) (pid count : Nat) : BlockState :=
  {mem:=fun r o => heap (r,o),regs:=fun _ _ _ => none,pids:=fun _ => pid,undef:=fun _ _ => 0,numPids:=fun _ => count}
abbrev Output (h : Call) := Fin h.cols ⊕ Unit
def address (h : Call) (row : Fin h.rows) : Output h → Address
  | .inl j => ("y",row.val*h.cols+j.val)
  | .inr _ => ("r",row.val)
def writes (h : Call) (row : Fin h.rows) (a : Address) : Prop :=
  (∃ j : Fin h.cols,a=(("y":RegionName),row.val*h.cols+j.val)) ∨ a=(("r":RegionName),row.val)
def reads (_h : Call) (_row : Fin _h.rows) (a : Address) : Prop := a.1="x" ∨ a.1="w"
def expected (h : Call) (heap : Heap) (row : Fin h.rows) : Output h → ℝ
  | .inl j => RealReference.output h.gemma h.eps
      (RealRows.raw h (entry heap row.val h.rows) "x" h.cols)
      (RealRows.raw h (entry heap row.val h.rows) "w" 0) j
  | .inr _ => RealReference.inv h.eps (RealRows.raw h (entry heap row.val h.rows) "x" h.cols)
def execute (h : Call) (row : Fin h.rows) (heap : Heap) : Option Heap :=
  (VeriTile.Triton.exec (Kernels.forward h).eraseDType.toAlgKernel (entry heap row.val h.rows)).map
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
    · exact (by decide : ("y":RegionName)≠"r") (congrArg Prod.fst (ha.symm.trans hb))
    · exact (by decide : ("r":RegionName)≠"y") (congrArg Prod.fst (ha.symm.trans hb))
    · exact ne (Fin.ext (congrArg Prod.snd (ha.symm.trans hb)))
  · intro hr
    rcases ha with ⟨j,rfl⟩|rfl <;> simp [reads] at hr

theorem step_complete (h : Call) (row : Fin h.rows) (heap : Heap) :
    ∃ final,execute h row heap=some final ∧
      (∀ i,read (final (address h row i))=expected h heap row i) ∧
      (∀ addr,¬writes h row addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hr,hf⟩ := RealRows.forward_complete h (entry heap row.val h.rows)
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

theorem expected_stable (h : Call) (row : Fin h.rows) (a b : Heap)
    (stable : ∀ addr,reads h row addr → a addr=b addr) (i : Output h) :
    expected h a row i=expected h b row i := by
  have x : RealRows.raw h (entry a row.val h.rows) "x" h.cols=
      RealRows.raw h (entry b row.val h.rows) "x" h.cols := by
    funext j
    exact congrArg read (stable ("x",row.val*h.cols+j.val) (Or.inl rfl))
  have w : RealRows.raw h (entry a row.val h.rows) "w" 0=
      RealRows.raw h (entry b row.val h.rows) "w" 0 := by
    funext j
    exact congrArg read (stable ("w",row.val*0+j.val) (Or.inr rfl))
  cases i <;> simp only [expected,x,w]

def run (h : Call) := GridRun.run (execute h) (List.finRange h.rows)

theorem complete (h : Call) (heap : Heap) :
    ∃ final,run h heap=some final ∧
      (∀ row : Fin h.rows,∀ i,read (final (address h row i))=expected h heap row i) ∧
      (∀ addr,(∀ row : Fin h.rows,¬writes h row addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute h) (fun _ => Output h) (address h) read
    (expected h) (reads h) (writes h) (inside h) (List.finRange h.rows) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete h w)
    (fun w _ => expected_stable h w) heap
  exact ⟨final,he,fun row => ho row (List.mem_finRange row),fun addr out => hf addr (fun row _ => out row)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridForward
