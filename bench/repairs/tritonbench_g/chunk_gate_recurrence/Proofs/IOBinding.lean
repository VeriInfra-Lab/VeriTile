import bench.repairs.tritonbench_g.chunk_gate_recurrence.IO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.IOBinding
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

theorem launch_head (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (u : State α) :
    ((IO.launch M kind c s seed p).initial u).pids 0=p.head := rfl
theorem launch_key (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (u : State α) :
    ((IO.launch M kind c s seed p).initial u).pids 1=p.key := rfl
theorem launch_value (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (u : State α) :
    ((IO.launch M kind c s seed p).initial u).pids 2=p.value := rfl

theorem forward_bounds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.forward M c s seed p).Inputs) (bounds : RegionName → Nat)
    (hb : (IO.forward M c s seed p).InBounds bounds a) :
    KernelSafety.ForwardBounds c s seed p bounds where
  source := fun t ht e j => hb.2.1 (0 : Fin 3) (⟨t,ht⟩,e,j,PUnit.unit) trivial
  gate := fun t ht => hb.2.1 (1 : Fin 3) (⟨t,ht⟩,PUnit.unit) trivial
  initial := fun hs e j => hb.2.1 (2 : Fin 3) (e,j,PUnit.unit) hs
  output := fun t ht e j => hb.2.2.1 (0 : Fin 1) (⟨t,ht⟩,e,j,PUnit.unit) trivial

theorem backward_bounds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.backward M c s seed p).Inputs) (bounds : RegionName → Nat)
    (hb : (IO.backward M c s seed p).InBounds bounds a) :
    KernelSafety.BackwardBounds c p bounds where
  saved := fun t ht e j => hb.2.1 (0 : Fin 3) (⟨t,ht⟩,e,j,PUnit.unit) trivial
  gate := fun t ht => hb.2.1 (1 : Fin 3) (⟨t,ht⟩,PUnit.unit) trivial
  incoming := fun t ht e j => hb.2.1 (2 : Fin 3) (⟨t,ht⟩,e,j,PUnit.unit) trivial
  gradient := fun t ht e j => hb.2.2.1 (0 : Fin 3) (⟨t,ht⟩,e,j,PUnit.unit) trivial
  gateGradient := fun t ht => hb.2.2.1 (1 : Fin 3) (⟨t,ht⟩,PUnit.unit) trivial
  seedGradient := fun e j => hb.2.2.1 (2 : Fin 3) (e,j,PUnit.unit) trivial

theorem safe (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (hn : 0<c.N) (a : (IO.program M kind c s seed p).Inputs)
    (bounds : RegionName → Nat) (hb : (IO.program M kind c s seed p).InBounds bounds a)
    (u : State α) : ScheduleSafe bounds ((IO.program M kind c s seed p).schedule a) u := by
  cases kind with
  | forward =>
    change ScheduleSafe bounds [IO.launch M .forward c s seed p] u
    rw [scheduleSafe_singleton]
    exact KernelSafety.forward_source M c s seed p bounds
      (forward_bounds M c s seed p a bounds hb) _ hn rfl rfl rfl
  | backward =>
    change ScheduleSafe bounds [IO.launch M .backward c s seed p] u
    rw [scheduleSafe_singleton]
    exact KernelSafety.backward_source M c s seed p bounds
      (backward_bounds M c s seed p a bounds hb) _ hn rfl rfl rfl

theorem frame (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.program M kind c s seed p).Inputs) (u v : State α)
    (hf : KernelMemory.Frame kind c p (IO.launch M kind c s seed p |>.initial u) v) :
    (IO.program M kind c s seed p).Frame a u v := by
  intro r off outside
  have excluded : ¬KernelMemory.footprint kind c p r off := by
    cases kind with
    | forward =>
      rintro ⟨rfl,t,e,j,rfl⟩
      exact (outside.1 (0 : Fin 1) (t,e,j,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)
    | backward =>
      rintro (⟨rfl,t,e,j,rfl⟩ | ⟨rfl,t,rfl⟩ | ⟨rfl,e,j,rfl⟩)
      · exact (outside.1 (0 : Fin 3) (t,e,j,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)
      · exact (outside.1 (1 : Fin 3) (t,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)
      · exact (outside.1 (2 : Fin 3) (e,j,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)
  simpa using hf r off excluded

theorem forward_eq (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.forward M c s seed p).Inputs) (u : State α)
    (hin : (IO.forward M c s seed p).InputBinding a u) (t : Nat) (ht : t<c.N) :
    KernelReference.forward M (KernelReference.read c s p u) seed t =
      KernelReference.forward M (Reference.forwardInputs a) seed t := by
  induction t with
  | zero =>
    funext i
    cases hs : seed
    · rfl
    ·
      have h := hin (2 : Fin 3) i hs
      simpa [KernelReference.forward,KernelReference.initial,KernelReference.read,
        Reference.forwardInputs,IO.forward,raw,hs,mapValue] using congrArg (M.cast (some .fp32) .real .real) h
  | succ t ih =>
    funext i
    have gate := hin (1 : Fin 3) (⟨t,by omega⟩,PUnit.unit) trivial
    have source := hin (0 : Fin 3) (⟨t,by omega⟩,i) trivial
    simp only [KernelReference.forward,ih (by omega)]
    congr 1
    · simpa [KernelReference.read,Reference.forwardInputs,IO.time,show t<c.N-1 by omega,
        IO.forward,mapValue] using gate
    · simpa [KernelReference.read,Reference.forwardInputs,IO.time,show t<c.N-1 by omega,
        IO.forward,raw,mapValue] using source

theorem reverse_eq (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.backward M c s seed p).Inputs) (u : State α)
    (hin : (IO.backward M c s seed p).InputBinding a u) (n : Nat) (hn : n≤c.N) :
    KernelReference.reverse M (KernelReference.read c s p u) c.N n =
      KernelReference.reverse M (Reference.backwardInputs a) c.N n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext i
    have gate := hin (1 : Fin 3) (⟨c.N-1-n,by omega⟩,PUnit.unit) trivial
    have incoming := hin (2 : Fin 3) (⟨c.N-1-n,by omega⟩,i) trivial
    simp only [KernelReference.reverse,ih (by omega)]
    congr 1
    · simpa [KernelReference.read,Reference.backwardInputs,IO.time,show c.N-1-n<c.N by omega,
        IO.backward,mapValue] using gate
    · simpa [KernelReference.read,Reference.backwardInputs,IO.time,show c.N-1-n<c.N by omega,
        IO.backward,raw,mapValue] using incoming

theorem saved_eq (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (a : (IO.backward M c s seed p).Inputs) (u : State α)
    (hin : (IO.backward M c s seed p).InputBinding a u) (t : Nat) (ht : t<c.N-1) :
    (KernelReference.read c s p u).saved t=(Reference.backwardInputs a).saved t := by
  funext i
  simpa [KernelReference.read,Reference.backwardInputs,IO.time,ht,IO.backward,raw,mapValue]
    using hin (0 : Fin 3) (⟨t,ht⟩,i) trivial

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.IOBinding
