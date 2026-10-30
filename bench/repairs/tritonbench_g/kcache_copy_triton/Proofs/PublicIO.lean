import bench.repairs.tritonbench_g.kcache_copy_triton.IO
import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Exactly the pre-existing destination injectivity condition, on logical
metadata rather than interpreter memory. Aliased input regions remain legal. -/
def OutputInjective {α : Type} (p : Parameters) (ids : Nat → Nat) (vals : Inputs α p) : Prop :=
  Function.Injective (outputIndex p ids (vals 0 ()) (vals 1 ()))

/-- The mathematical payload formula is the identity copy. -/
def formula {α : Type} (x : α) := x

def target {α : Type} (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (f : α → α) (vals : (io M v p ids).Inputs) : (io M v p ids).Outputs :=
  fun o => Fin.cases (fun j => f (vals 2 j)) (fun z => Fin.elim0 z) o

@[simp] theorem start_pids {α : Type} (M : Algebra α) (v : Bool) (p : Parameters)
    (ids : Nat → Nat) (s : State α) : (start M v p ids s).pids = ids := rfl
@[simp] theorem start_mem {α : Type} (M : Algebra α) (v : Bool) (p : Parameters)
    (ids : Nat → Nat) (s : State α) : (start M v p ids s).mem = s.mem := rfl

variable {α : Type} [Inhabited α]

private theorem position_eq (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (vals : (io M v p ids).Inputs) (s : State α) (hin : (io M v p ids).InputBinding vals s) :
    KernelFP.position p (start M v p ids s) = position p ids (vals 0 ()) := by
  have h := hin 0 () trivial
  change (s.mem p.lengths.cast (sequenceIndex p ids).toNat).read .int = vals 0 () at h
  change (s.mem p.lengths.cast (sequenceIndex p ids).toNat).read .int + _ = _
  rw [h]
  rfl

private theorem block_eq (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (vals : (io M v p ids).Inputs) (s : State α) (hin : (io M v p ids).InputBinding vals s) :
    KernelFP.selectedBlock p (start M v p ids s) = vals 1 () := by
  unfold KernelFP.selectedBlock
  rw [position_eq M v p ids vals s hin]
  exact hin 1 () trivial

private theorem output_eq (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (vals : (io M v p ids).Inputs) (s : State α) (hin : (io M v p ids).InputBinding vals s) :
    KernelFP.targetOffset p (start M v p ids s) = outputIndex p ids (vals 0 ()) (vals 1 ()) := by
  funext i
  unfold KernelFP.targetOffset
  rw [position_eq M v p ids vals s hin,block_eq M v p ids vals s hin]
  rfl

theorem schedule_safe (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (vals : (io M v p ids).Inputs) (bounds : RegionName → Nat)
    (hb : (io M v p ids).InBounds bounds vals) (s : State α)
    (hin : (io M v p ids).InputBinding vals s) :
    ScheduleSafe bounds ((io M v p ids).schedule vals) s := by
  rw [scheduleSafe_singleton]
  apply KernelSafety.trace_safe
  · exact hb.2.1 0 () trivial
  · rw [position_eq M v p ids vals s hin]
    exact hb.2.1 1 () trivial
  · intro i
    exact hb.2.1 2 i trivial
  · rw [output_eq M v p ids vals s hin]
    intro i
    exact hb.2.2.1 0 i trivial

theorem frame (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (vals : (io M v p ids).Inputs) (s : State α) (hin : (io M v p ids).InputBinding vals s) :
    (io M v p ids).Frame vals s (KernelFP.execution M v p (start M v p ids s)).val := by
  intro r o outside
  apply KernelFP.frame
  by_cases hr : r = p.KCache
  · right
    intro i
    rw [output_eq M v p ids vals s hin]
    exact (outside.1 0 i trivial).resolve_left (fun h => h hr)
  · exact Or.inl hr

theorem implements (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat)
    (f : α → α) (load : ∀ x, M.fp32Load x = f x) :
    (io M v p ids).ImplementsOn (OutputInjective p ids) (target M v p ids f) := by
  intro vals injective bounds hb s hin
  refine ⟨schedule_safe M v p ids vals bounds hb s hin,
    (KernelFP.execution M v p (start M v p ids s)).val,?_,?_,frame M v p ids vals s hin⟩
  · rw [runSchedule_singleton]
    exact KernelFP.execute M v p (start M v p ids s)
  · intro o i _
    fin_cases o
    have injective' : Function.Injective (KernelFP.targetOffset p (start M v p ids s)) := by
      rw [output_eq M v p ids vals s hin]
      exact injective
    have cells := State.scatter_readback p.KCache
      (KernelFP.targetOffset p (start M v p ids s))
      (fun j => Cell.mk .real (M.fp32Load
        (((start M v p ids s).mem p.K (KernelFP.sourceOffset p ids j)).read .real)))
      (start M v p ids s) i injective'
    change (KernelFP.copied M p (start M v p ids s)).mem p.KCache
      (KernelFP.targetOffset p (start M v p ids s) i) = _ at cells
    rw [←KernelFP.execution_memory M v p (start M v p ids s)] at cells
    rw [output_eq M v p ids vals s hin] at cells
    have hinput := hin 2 i trivial
    change (s.mem p.K (inputIndex p ids i)).read .real = vals 2 i at hinput
    change mapCell id ((KernelFP.execution M v p (start M v p ids s)).val.mem p.KCache
      (outputIndex p ids (vals 0 ()) (vals 1 ()) i)) = Cell.mk .real (f (vals 2 i))
    rw [cells]
    change Cell.mk .real (M.fp32Load ((s.mem p.K (inputIndex p ids i)).read .real)) = _
    rw [hinput,load]

/-- No injectivity or finite-payload assumption is needed for equivalence:
even colliding destinations produce exactly the same snapshot scatter. -/
theorem equivalent (M : Algebra α) (p : Parameters) (ids : Nat → Nat) :
    (io M Bool.false p ids).Equivalent (candidate M p ids) := by
  intro vals _ bounds hb s hin
  refine ⟨schedule_safe M Bool.false p ids vals bounds hb s hin,
    schedule_safe M Bool.true p ids vals bounds hb s hin,
    (KernelFP.execution M Bool.false p (start M Bool.false p ids s)).val,
    (KernelFP.execution M Bool.true p (start M Bool.true p ids s)).val,
    ?_,?_,?_,
    frame M Bool.false p ids vals s hin,frame M Bool.true p ids vals s hin⟩
  · rw [runSchedule_singleton]
    exact KernelFP.execute M Bool.false p (start M Bool.false p ids s)
  · rw [runSchedule_singleton]
    exact KernelFP.execute M Bool.true p (start M Bool.true p ids s)
  · intro o i _
    rw [KernelFP.execution_memory,KernelFP.execution_memory]
    rfl

#axiomsClean schedule_safe
#axiomsClean implements
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
