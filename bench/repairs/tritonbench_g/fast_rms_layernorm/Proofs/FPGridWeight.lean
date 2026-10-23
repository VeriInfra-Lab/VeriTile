import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPGridForward
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightMemory

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridWeight
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Geometry
open FPGridForward (Address Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

abbrev Output (h : Call) (pid : Fin (featureCount h)) := {j : Fin 256 // column pid.val j<h.cols}
def address (h : Call) (pid : Fin (featureCount h)) (j : Output h pid) : Address := ("dw",column pid.val j.val)
def writes (h : Call) (pid : Fin (featureCount h)) (a : Address) : Prop := ∃ j : Output h pid,a=address h pid j
def reads (_h : Call) (_pid : Fin (featureCount _h)) (a : Address) : Prop := a.1="x" ∨ a.1="dx" ∨ a.1="r"
def expected (M : Algebra α) (h : Call) (heap : Heap α) (pid : Fin (featureCount h)) (j : Output h pid) : Cell α :=
  FPWeightReference.output M h.gemma
    (fun row => FPWeightStep.raw M h 256 pid.val (entry heap pid.val (featureCount h)) "x" row (j.val,PUnit.unit))
    (fun row => FPWeightStep.raw M h 256 pid.val (entry heap pid.val (featureCount h)) "dx" row (j.val,PUnit.unit))
    (FPWeightStep.saved (entry heap pid.val (featureCount h))) h.rows
def execute (M : Algebra α) (h : Call) (pid : Fin (featureCount h)) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (engine M) (Kernels.weightBackward h 256) (entry heap pid.val (featureCount h))).map
    (fun v a => v.mem a.1 a.2)

theorem separate (h : Call) (a b : Fin (featureCount h)) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  constructor
  · rintro ⟨j,hb⟩
    obtain ⟨i,ha⟩:=ha
    have eq := congrArg Prod.snd (ha.symm.trans hb)
    exact ne (Fin.ext (column_injective a.val b.val i.val j.val eq).1)
  · intro hr
    obtain ⟨j,rfl⟩:=ha
    simp [reads,address] at hr

theorem step_complete (M : Algebra α) (h : Call) (pid : Fin (featureCount h)) (heap : Heap α) :
    ∃ final,execute M h pid heap=some final ∧
      (∀ j,final (address h pid j)=expected M h heap pid j) ∧
      (∀ addr,¬writes h pid addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hf⟩ := FPWeightMemory.complete M h 256 (entry heap pid.val (featureCount h))
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],fun j => hy j.val j.property,?_⟩
  rintro ⟨q,o⟩ outside
  apply hf
  rintro ⟨same,j,active,eq⟩
  exact outside ⟨⟨j,active⟩,Prod.ext same eq⟩

theorem expected_stable (M : Algebra α) (h : Call) (pid : Fin (featureCount h)) (a b : Heap α)
    (stable : ∀ addr,reads h pid addr → a addr=b addr) (j : Output h pid) :
    expected M h a pid j=expected M h b pid j := by
  have hx : (fun row => FPWeightStep.raw M h 256 pid.val (entry a pid.val (featureCount h)) "x" row (j.val,PUnit.unit))=
      (fun row => FPWeightStep.raw M h 256 pid.val (entry b pid.val (featureCount h)) "x" row (j.val,PUnit.unit)) := by
    funext row
    simp only [FPWeightStep.raw,entry]
    rw [stable _ (Or.inl rfl)]
  have hg : (fun row => FPWeightStep.raw M h 256 pid.val (entry a pid.val (featureCount h)) "dx" row (j.val,PUnit.unit))=
      (fun row => FPWeightStep.raw M h 256 pid.val (entry b pid.val (featureCount h)) "dx" row (j.val,PUnit.unit)) := by
    funext row
    simp only [FPWeightStep.raw,entry]
    rw [stable _ (Or.inr (Or.inl rfl))]
  have hr : FPWeightStep.saved (entry a pid.val (featureCount h))=FPWeightStep.saved (entry b pid.val (featureCount h)) := by
    funext row
    exact congrArg (Cell.read .real) (stable ("r",row) (Or.inr (Or.inr rfl)))
  unfold expected
  rw [hx,hg,hr]

def run (M : Algebra α) (h : Call) := GridRun.run (execute M h) (List.finRange (featureCount h))

theorem complete (M : Algebra α) (h : Call) (heap : Heap α) :
    ∃ final,run M h heap=some final ∧
      (∀ pid : Fin (featureCount h),∀ j,final (address h pid j)=expected M h heap pid j) ∧
      (∀ addr,(∀ pid : Fin (featureCount h),¬writes h pid addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute M h) (Output h) (address h) id
    (expected M h) (reads h) (writes h) (fun pid j => ⟨j,rfl⟩) (List.finRange (featureCount h)) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete M h w)
    (fun w _ => expected_stable M h w) heap
  exact ⟨final,he,fun pid => ho pid (List.mem_finRange pid),fun addr out => hf addr (fun pid _ => out pid)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridWeight
