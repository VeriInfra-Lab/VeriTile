import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealGridForward
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightMemory

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridWeight
open VeriTile.Triton Geometry
open RealGridForward (Address Heap entry read)
noncomputable section

abbrev Output (h : Call) (pid : Fin (featureCount h)) := {j : Fin 256 // column pid.val j<h.cols}
def address (h : Call) (pid : Fin (featureCount h)) (j : Output h pid) : Address := ("dw",column pid.val j.val)
def writes (h : Call) (pid : Fin (featureCount h)) (a : Address) : Prop := ∃ j : Output h pid,a=address h pid j
def reads (_h : Call) (_pid : Fin (featureCount _h)) (a : Address) : Prop := a.1="x" ∨ a.1="dx" ∨ a.1="r"
def expected (h : Call) (heap : Heap) (pid : Fin (featureCount h)) (j : Output h pid) : ℝ :=
  RealWeightLoop.carry h 256 pid.val (entry heap pid.val (featureCount h)) h.rows j.val
def execute (h : Call) (pid : Fin (featureCount h)) (heap : Heap) : Option Heap :=
  (VeriTile.Triton.exec (RealWeightProgram.source h 256) (entry heap pid.val (featureCount h))).map
    (fun v a => v.mem a.1 a.2)

theorem separate (h : Call) (a b : Fin (featureCount h)) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  constructor
  · rintro ⟨j,hb⟩
    obtain ⟨i,ha⟩:=ha
    exact ne (Fin.ext (column_injective a.val b.val i.val j.val (congrArg Prod.snd (ha.symm.trans hb))).1)
  · intro hr
    obtain ⟨j,rfl⟩:=ha
    simp [reads,address] at hr

theorem step_complete (h : Call) (pid : Fin (featureCount h)) (heap : Heap) :
    ∃ final,execute h pid heap=some final ∧
      (∀ j,read (final (address h pid j))=expected h heap pid j) ∧
      (∀ addr,¬writes h pid addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hf⟩ := RealWeightMemory.complete h 256 (entry heap pid.val (featureCount h))
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],fun j => hy j.val j.property,?_⟩
  rintro ⟨q,o⟩ outside
  apply hf
  rintro ⟨same,j,active,eq⟩
  exact outside ⟨⟨j,active⟩,Prod.ext same eq⟩

theorem expected_stable (h : Call) (pid : Fin (featureCount h)) (a b : Heap)
    (stable : ∀ addr,reads h pid addr → a addr=b addr) (j : Output h pid) :
    expected h a pid j=expected h b pid j := by
  unfold expected RealWeightLoop.carry
  apply Finset.sum_congr rfl
  intro row _
  have hx:=congrArg read (stable ("x",row*h.cols+column pid.val j.val) (Or.inl rfl))
  have hg:=congrArg read (stable ("dx",row*h.cols+column pid.val j.val) (Or.inr (Or.inl rfl)))
  have hr:=congrArg read (stable ("r",row) (Or.inr (Or.inr rfl)))
  change (if column pid.val j.val<h.cols then read (a ("dx",row*h.cols+column pid.val j.val)) else 0)*
      ((if column pid.val j.val<h.cols then read (a ("x",row*h.cols+column pid.val j.val)) else 0)*read (a ("r",row)))=_
  rw [hx,hg,hr]
  rfl

def run (h : Call) := GridRun.run (execute h) (List.finRange (featureCount h))

theorem complete (h : Call) (heap : Heap) :
    ∃ final,run h heap=some final ∧
      (∀ pid : Fin (featureCount h),∀ j,read (final (address h pid j))=expected h heap pid j) ∧
      (∀ addr,(∀ pid : Fin (featureCount h),¬writes h pid addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute h) (Output h) (address h) read
    (expected h) (reads h) (writes h) (fun pid j => ⟨j,rfl⟩) (List.finRange (featureCount h)) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete h w)
    (fun w _ => expected_stable h w) heap
  exact ⟨final,he,fun pid => ho pid (List.mem_finRange pid),fun addr out => hf addr (fun pid _ => out pid)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridWeight
