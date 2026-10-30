import bench.tritonbench_g.triton_argmax.Proofs.JITNativeIO
import bench.tritonbench_g.triton_argmax.Proofs.JITSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

variable (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c)
include g

def input (vals : ChannelValues Word [⟨.real,[c.size],BoundSource.X⟩,
    ⟨.int,[c.midBlock],Region.cast BoundSource.I⟩]) (i : Nat) : Word :=
  if h : i < c.size then vals 0 (⟨i,h⟩,PUnit.unit) else 0

/-- The original initialized INT64 gather window, including its padded tail. -/
def initialized (index64 : Bool)
    (vals : (JITIO.nativeGlobal casts kind binding rest c index64).Inputs) : Prop :=
  ∀ j : Fin c.midBlock, vals 1 (j,PUnit.unit) = (0 : Int)

def mathematical (positive : 0 < c.block) (midPositive : 0 < c.midBlock) (index64 : Bool)
    (vals : ChannelValues Word [⟨.real,[c.size],BoundSource.X⟩,
      ⟨.int,[c.midBlock],Region.cast BoundSource.I⟩]) :
    ChannelValues Word [⟨.int,[1],Region.cast BoundSource.O⟩] :=
  Fin.cases (fun _ => Int.ofNat (globalReference (Request.scratch casts kind) c positive midPositive
    (input c vals))) (fun i => Fin.elim0 i)

theorem first_safe (s : State Word) (pid : Nat) (index64 : Bool)
    (positive : 0 < c.block) (active : pid * c.block < c.size)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i, i < c.size → i < bounds BoundSource.X)
    (valueBound : pid < bounds BoundSource.V)
    (indexBound : pid < bounds (Region.cast BoundSource.I)) :
    (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0 pid).Safe bounds s := by
  let initial := { s with pids := fun ax => if ax = 0 then pid else s.pids ax }
  have selected : Typed.firstIndex initial BoundSource.X c.size c.block 0 positive < bounds BoundSource.X := by
    rw [TypedNative.first_index initial BoundSource.X c g 0 positive]
    simp only [initial, ite_true, Nat.add_zero]
    exact inputBounds _ (first_valid c g _ pid positive active)
  unfold Invocation.Safe
  rw [JITIO.firstLaunch_initial, JITIO.atProgram_initial]
  cases binding with
  | floating k round codec =>
    exact JITSafety.native_first k round rest initial BoundSource.X BoundSource.V BoundSource.I
      c.size c.block 0 index64 positive bounds inputBounds selected
      (by simpa [initial] using valueBound) (by simpa [initial] using indexBound)
  | integer k zero =>
    exact JITSafety.integer_first k (casts.integer k) rest initial BoundSource.X BoundSource.V BoundSource.I
      c.size c.block 0 index64 positive bounds inputBounds selected
      (by simpa [initial] using valueBound) (by simpa [initial] using indexBound)

theorem second_safe (s : State Word) (positive : 0 < c.midBlock)
    (bounds : RegionName → Nat)
    (valueBounds : ∀ i, i < c.blocks → i < bounds BoundSource.V)
    (indexBounds : ∀ i, i < c.midBlock → i < bounds (Region.cast BoundSource.I))
    (outputBound : 0 < bounds (Region.cast BoundSource.O)) :
    (JITIO.secondLaunch (JITBinding.ops binding rest) (JITBinding.second binding c) c.blocks).Safe bounds s := by
  unfold Invocation.Safe
  rw [JITIO.secondLaunch_initial]
  cases binding with
  | floating k round codec =>
    apply JITSafety.native_second k round rest s BoundSource.V BoundSource.I BoundSource.O
      c.blocks c.midBlock positive bounds valueBounds _ outputBound
    unfold TypedScratch.secondIndex
    rw [TypedReduction.selected_eq positive _ c.secondTree
      (g.secondNumeric ▸ g.second.numericComplete)]
    exact indexBounds _ (c.secondTree.pick _ _ _).isLt
  | integer k zero =>
    apply JITSafety.integer_second k (JITBinding.ops (.integer k zero) rest) s
      BoundSource.V BoundSource.I BoundSource.O c.blocks c.midBlock bounds valueBounds _ outputBound
    unfold JITInteger.secondIndex
    rw [IntegerReduction.shared_tree positive _ c.secondTree
      (g.secondNumeric ▸ g.second.numericComplete)]
    exact indexBounds _ (c.secondTree.pick _ _ _).isLt

theorem schedule_safe (s : State Word) (index64 : Bool)
    (positive : 0 < c.block) (midPositive : 0 < c.midBlock)
    (active : ∀ pid, pid < c.blocks → pid * c.block < c.size)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i, i < c.size → i < bounds BoundSource.X)
    (valueBounds : ∀ i, i < c.blocks → i < bounds BoundSource.V)
    (indexWrites : ∀ i, i < c.blocks → i < bounds (Region.cast BoundSource.I))
    (indexReads : ∀ i, i < c.midBlock → i < bounds (Region.cast BoundSource.I))
    (outputBound : 0 < bounds (Region.cast BoundSource.O)) :
    ScheduleSafe bounds (JITIO.nativeSchedule casts kind binding rest c index64) s := by
  rw [JITIO.nativeSchedule, scheduleSafe_append]
  constructor
  · have gridSafe : ∀ (jobs : List Nat), (∀ pid, pid ∈ jobs → pid < c.blocks) →
        ∀ before, ScheduleSafe bounds
          (jobs.map (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0)) before := by
      intro jobs
      induction jobs with
      | nil => intro _ _; trivial
      | cons pid jobs ih =>
        intro valid before
        refine ⟨first_safe casts kind binding rest c g before pid index64 positive
          (active pid (valid pid (by simp))) bounds inputBounds
          (valueBounds pid (valid pid (by simp))) (indexWrites pid (valid pid (by simp))), ?_⟩
        intro next _
        exact ih (fun p hp => valid p (by simp [hp])) next
    exact gridSafe _ (fun pid hp => List.mem_range.mp hp) s
  · intro middle _
    rw [scheduleSafe_singleton]
    exact second_safe casts kind binding rest c g middle midPositive bounds valueBounds indexReads outputBound

/-- Source safety and complete native output are independent obligations.
The only data domain is the original scratch initialization. -/
theorem implements (positive : 0 < c.block) (midPositive : 0 < c.midBlock)
    (active : ∀ pid, pid < c.blocks → pid * c.block < c.size) (index64 : Bool) :
    (JITIO.nativeGlobal casts kind binding rest c index64).ImplementsOn
      (initialized casts kind binding rest c index64)
      (mathematical casts kind c positive midPositive index64) := by
  intro vals zero bounds bounded s bindingInput
  let xs := input c vals
  have reads : ∀ i, i < c.size → (s.mem BoundSource.X i).read .real = xs i := by
    intro i hi
    have h := bindingInput 0 (⟨i,hi⟩,PUnit.unit) trivial
    simpa [JITIO.nativeGlobal, mapValue, xs, input, hi] using h
  have initial : ∀ i, i < c.midBlock → (s.mem (Region.cast BoundSource.I) i).read .int = 0 := by
    intro i hi
    exact (bindingInput 1 (⟨i,hi⟩,PUnit.unit) trivial).trans (zero ⟨i,hi⟩)
  refine ⟨schedule_safe casts kind binding rest c g s index64 positive midPositive active bounds
    (fun i hi => bounded.2.1 0 (⟨i,hi⟩,PUnit.unit) trivial)
    (fun i hi => bounded.2.2.2 0 (⟨i,hi⟩,PUnit.unit) trivial)
    (fun i hi => bounded.2.2.2 1 (⟨i,hi⟩,PUnit.unit) trivial)
    (fun i hi => bounded.2.1 1 (⟨i,hi⟩,PUnit.unit) trivial)
    (bounded.2.2.1 0 (⟨0,by decide⟩,PUnit.unit) trivial), ?_⟩
  obtain ⟨final, ran, output, frame⟩ := global_execution casts kind binding rest c g xs s
    index64 positive midPositive active reads initial
  refine ⟨final, ran, ?_, ?_⟩
  · intro o j _
    have ho : o = 0 := Fin.eq_zero o
    subst o
    change mapCell id (final.mem (Region.cast BoundSource.O) 0) = _
    rw [output]
    rfl
  · intro r o outside
    apply frame r o
    · exact outside.1 0 (⟨0,by decide⟩,PUnit.unit) trivial
    · intro pid hp
      exact ⟨outside.2 0 (⟨pid,hp⟩,PUnit.unit) trivial,
        outside.2 1 (⟨pid,hp⟩,PUnit.unit) trivial⟩

#axiomsClean implements
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeIO
