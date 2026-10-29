import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Index

namespace VeriTile.Bench.Repairs.UniformSampling.Native
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
open IndexProof
noncomputable section
open scoped Classical

/-- Source registers for one evaluated field/lane. Index arithmetic is widened
before multiplication; all four affine values are evaluated before masks. -/
structure Registers (α : Type) where
  localIndex : Nat
  pointerIndex : Nat
  active : Bool
  word : Random.Word
  uniform : α
  width : α
  value : α

def registers (M : Arithmetic α) (scalars : Scalars) (seed start n : Nat)
    (x : Location (block n)) : Registers α :=
  let localIndex:=outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2
  let word:=sourceWord (signed64 seed) (signed64 (start+x.1.done))
    x.2.1 (block n) x.2.2.1 x.2.2.2
  let u:=M.uniform32 word
  let width:=M.sub32 (M.extract scalars.upper) (M.extract scalars.lower)
  ⟨localIndex,min localIndex (chunkSize n x.1-1),decide (localIndex<chunkSize n x.1),
    word,u,width,M.fma32 u width (M.extract scalars.lower)⟩

def step (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars) (seed start : Nat)
    (scratch : Scratch n) (m : Store) (x : Location (block n)) : Store :=
  let regs:=registers M scalars seed start n x
  if regs.active then api.write m (scratchAddress scratch (4*x.1.done+regs.pointerIndex)) regs.value else m

def run (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars) (seed start : Nat)
    (scratch : Scratch n) (xs : List (Location (block n))) (m : Store) : Store :=
  xs.foldl (step api M scalars seed start scratch) m

def events (M : Arithmetic α) (scalars : Scalars) (seed start : Nat)
    (scratch : Scratch n) (xs : List (Location (block n))) : List (Write α) :=
  xs.filterMap fun x=>
    if outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<chunkSize n x.1 then
      some ⟨scratchAddress scratch (globalIndex x),
        affine M scalars (sourceWord (signed64 seed) (signed64 (start+x.1.done))
          x.2.1 (block n) x.2.2.1 x.2.2.2)⟩
    else none

theorem fold_events (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (xs : List (Location (block n))) (m : Store) :
    run api M scalars seed start scratch xs m=Memory.run api (events M scalars seed start scratch xs) m := by
  induction xs generalizing m with
  |nil=>rfl
  |cons x xs ih=>
    simp only [run,List.foldl_cons]
    rw [show xs.foldl (step api M scalars seed start scratch)
      (step api M scalars seed start scratch m x)=
        Memory.run api (events M scalars seed start scratch xs)
          (step api M scalars seed start scratch m x) from ih _]
    by_cases h:outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<chunkSize n x.1
    · have hs: min (outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2)
        (chunkSize n x.1-1)=outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2:=Nat.min_eq_left (by omega)
      simp [events,List.filterMap_cons,h,Memory.run,step,registers,hs,globalIndex,affine]
    · simp [events,List.filterMap_cons,h,Memory.run,step,registers]

theorem mem_events {M : Arithmetic α} {scalars : Scalars} {seed start : Nat}
    {scratch : Scratch n} {xs : List (Location (block n))} {e : Write α} :
    e∈events M scalars seed start scratch xs ↔
      ∃ x,x∈xs ∧ outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<chunkSize n x.1 ∧
        e=⟨scratchAddress scratch (globalIndex x),
          affine M scalars (sourceWord (signed64 seed) (signed64 (start+x.1.done))
            x.2.1 (block n) x.2.2.1 x.2.2.2)⟩ := by
  simp only [events,List.mem_filterMap]
  constructor
  · rintro ⟨x,hx,h⟩
    split at h
    · exact ⟨x,hx,‹_›,(Option.some.inj h).symm⟩
    · simp at h
  · rintro ⟨x,hx,ha,rfl⟩
    exact ⟨x,hx,by simp [ha]⟩

def result (M : Arithmetic α) (scalars : Scalars) (seed start B i : Nat) : α :=
  affine M scalars (Spec.word seed start B i)

theorem scratch_apart (scratch : Scratch n) {i j : Nat} (h:i≠j) :
    Apart (scratchAddress scratch i) (scratchAddress scratch j) := by
  unfold scratchAddress Apart
  omega

theorem readback (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (m : Store) (hs:block n∣start)
    {i : Nat} (hi:i<n) :
    api.read (run api M scalars seed start scratch (schedule n start) m) (scratchAddress scratch i)=
      result M scalars seed start (block n) i := by
  rw [fold_events]
  apply run_readback
  · obtain ⟨x,hx,he,ha⟩:=coverage hi hs
    refine ⟨⟨scratchAddress scratch i,result M scalars seed start (block n) i⟩,?_,rfl,rfl⟩
    apply mem_events.mpr
    refine ⟨x,hx,ha,?_⟩
    rw [word_binding hs hx,he]
    rfl
  · intro e he
    obtain ⟨x,hx,ha,rfl⟩:=mem_events.mp he
    by_cases h:globalIndex x=i
    · left
      rw [word_binding hs hx,h]
      exact ⟨rfl,rfl⟩
    · right
      exact scratch_apart scratch h

/-- Any legal lane serialization has exactly the same readback: all active
writes are unique in value/address, and every logical index is covered. -/
theorem serialized_readback (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (xs : List (Location (block n)))
    (same:∀ x,x∈xs↔x∈schedule n start) (m : Store) (hs:block n∣start)
    {i : Nat} (hi:i<n) :
    api.read (run api M scalars seed start scratch xs m) (scratchAddress scratch i)=
      result M scalars seed start (block n) i := by
  rw [fold_events]
  apply run_readback
  · obtain ⟨x,hx,he,ha⟩:=coverage hi hs
    refine ⟨⟨scratchAddress scratch i,result M scalars seed start (block n) i⟩,?_,rfl,rfl⟩
    apply mem_events.mpr
    refine ⟨x,(same x).mpr hx,ha,?_⟩
    rw [word_binding hs hx,he]
    rfl
  · intro e he
    obtain ⟨x,hx,ha,rfl⟩:=mem_events.mp he
    by_cases h:globalIndex x=i
    · left
      rw [word_binding hs ((same x).mp hx),h]
      exact ⟨rfl,rfl⟩
    · right
      exact scratch_apart scratch h

theorem byte_frame (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (xs : List (Location (block n))) (m : Store)
    (hs:block n∣start) (sub:∀ x,x∈xs→x∈schedule n start) (b : Nat)
    (outside:∀ i,i<n→b<scratchAddress scratch i ∨ scratchAddress scratch i+4≤b) :
    api.byte (run api M scalars seed start scratch xs m) b=api.byte m b := by
  rw [fold_events]
  apply run_byte_frame
  intro e he
  obtain ⟨x,hx,ha,rfl⟩:=mem_events.mp he
  exact outside (globalIndex x) ((mask_global hs (sub x hx)).mp ha)

#axiomsClean fold_events
#axiomsClean readback
#axiomsClean serialized_readback
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Native
