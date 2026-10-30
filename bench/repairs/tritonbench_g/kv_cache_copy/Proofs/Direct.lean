import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Source

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section
set_option maxHeartbeats 4000000

abbrev Byte := Fin 256

def observeByte (heap : Memory Word) (byte : Nat) : Byte :=
  ⟨(heap (4*(byte/4))).toNat/256^(byte%4)%256,Nat.mod_lt _ (by decide)⟩

def Touched (m : Inputs) (p : Config) (l : Lengths) (t : Tables) (byte : Nat) : Prop :=
  ∃ key s head d,s<p.count ∧ head<p.heads ∧ d<p.dims ∧
    address m (outputRole key) (outputIndex p l t key s head d)≤byte ∧
    byte<address m (outputRole key) (outputIndex p l t key s head d)+4

theorem byte_frame (m : Inputs) (p : Config) (l : Lengths) (t : Tables)
    (before after : Memory Word) (copy : CacheCopy m p l t before after)
    (byte : Nat) (outside : ¬Touched m p l t byte) : observeByte after byte=observeByte before byte := by
  have frame := copy.2 (4*(byte/4)) (by
    intro key s head d hs hh hd same
    apply outside
    refine ⟨key,s,head,d,hs,hh,hd,?_,?_⟩
    all_goals omega)
  simp only [observeByte,frame]

theorem word_alignment (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e : Lane) (he : ValidLane (config m b) e) : targetAddress m (config m b) l t e%4=0 := by
  have bound := target_bound m b l t h e he
  have positive : 0<(m (outputRole e.key)).numel := by unfold Inside at bound;omega
  simp [targetAddress,address,Nat.add_mod,h.alignment _ positive]

theorem store_bytes_disjoint (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e f : Lane) (he : ValidLane (config m b) e) (hf : ValidLane (config m b) f) (different : e≠f) :
    targetAddress m (config m b) l t e+4≤targetAddress m (config m b) l t f ∨
    targetAddress m (config m b) l t f+4≤targetAddress m (config m b) l t e := by
  have alignedE := word_alignment m b l t h e he
  have alignedF := word_alignment m b l t h f hf
  have distinct : targetAddress m (config m b) l t e≠targetAddress m (config m b) l t f :=
    fun eq => different (target_injective m b l t h e f he hf eq)
  omega

namespace Direct
structure Result where
  returned : Option Unit
  (keyObject valueObject : Nat)
  (keyMetadata valueMetadata : VeriTile.Torch.TensorMetadata)
  memory : Memory Word

def runWrapper (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (keyObject valueObject : Nat)
    (before : Memory Word) : Except InputError Result := do
  checkInputs m b l t
  return ⟨none,keyObject,valueObject,m .keyCache,m .valueCache,Native.launch m (config m b) id before⟩

theorem correct (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (keyObject valueObject : Nat) (before : Memory Word) (binding : TensorBinding m b l t id before) :
    ∃ out,runWrapper m b l t keyObject valueObject before=.ok out ∧ out.returned=none ∧
      out.keyObject=keyObject ∧ out.valueObject=valueObject ∧ out.keyMetadata=m .keyCache ∧
      out.valueMetadata=m .valueCache ∧ CacheCopy m (config m b) l t before out.memory ∧
      (∀ byte,¬Touched m (config m b) l t byte → observeByte out.memory byte=observeByte before byte) := by
  have execution := Native.launch_exact m b l t h id before binding
  have copy := launch_correct m b l t h before
  refine ⟨⟨none,keyObject,valueObject,m .keyCache,m .valueCache,launch m (config m b) l t before⟩,
    ?_,rfl,rfl,rfl,rfl,rfl,copy,?_⟩
  · simp [runWrapper,(checkInputs_iff m b l t).mpr h,execution]
  · exact byte_frame m (config m b) l t before _ copy

#axiomsClean correct
end Direct
#axiomsClean store_bytes_disjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy
