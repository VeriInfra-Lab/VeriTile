import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Source
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
set_option maxHeartbeats 4000000

abbrev Byte := Fin 256

def observeByte (heap : Memory Word) (byte : Nat) : Byte :=
  ⟨(heap (4*(byte/4))).toNat/256^(byte%4)%256,Nat.mod_lt _ (by decide)⟩

def Touched (m : Inputs) (p : Config) (l : Lengths) (t : Tables) (byte : Nat) : Prop :=
  ∃ s head d,s<p.count ∧ head<p.heads ∧ d<p.dims ∧
    address m (.cache) (outputIndex p l t s head d)≤byte ∧
    byte<address m (.cache) (outputIndex p l t s head d)+4

theorem byte_frame (m : Inputs) (p : Config) (l : Lengths) (t : Tables)
    (before after : Memory Word) (copy : CacheCopy m p l t before after)
    (byte : Nat) (outside : ¬Touched m p l t byte) : observeByte after byte=observeByte before byte := by
  have frame := copy.2 (4*(byte/4)) (by
    intro s head d hs hh hd same
    apply outside
    refine ⟨s,head,d,hs,hh,hd,?_,?_⟩
    all_goals omega)
  simp only [observeByte,frame]

theorem word_alignment (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e : Lane) (he : ValidLane (config m n b) e) : targetAddress m (config m n b) l t e%4=0 := by
  have bound := target_bound m n b l t h e he
  have positive : 0<(m .cache).numel := by unfold Inside at bound;omega
  simp [targetAddress,address,Nat.add_mod,h.alignment _ positive]

theorem store_bytes_disjoint (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e f : Lane) (he : ValidLane (config m n b) e) (hf : ValidLane (config m n b) f) (different : e≠f) :
    targetAddress m (config m n b) l t e+4≤targetAddress m (config m n b) l t f ∨
    targetAddress m (config m n b) l t f+4≤targetAddress m (config m n b) l t e := by
  have alignedE := word_alignment m n b l t h e he
  have alignedF := word_alignment m n b l t h f hf
  have distinct : targetAddress m (config m n b) l t e≠targetAddress m (config m n b) l t f :=
    fun eq => different (target_injective m n b l t h e f he hf eq)
  omega

namespace Direct
structure Result where
  returned : Option Unit
  cacheObject : Nat
  cacheMetadata : VeriTile.Torch.TensorMetadata
  memory : Memory Word

def runWrapper (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (cacheObject : Nat)
    (before : Memory Word) : Except InputError Result := do
  checkInputs m n b l t
  return ⟨none,cacheObject,m .cache,Native.launch m (config m n b) id before⟩

theorem correct (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (cacheObject : Nat) (before : Memory Word) (binding : TensorBinding m n b l t id before) :
    ∃ out,runWrapper m n b l t cacheObject before=.ok out ∧ out.returned=none ∧
      out.cacheObject=cacheObject ∧ out.cacheMetadata=m .cache ∧ CacheCopy m (config m n b) l t before out.memory ∧
      (∀ byte,¬Touched m (config m n b) l t byte → observeByte out.memory byte=observeByte before byte) := by
  have execution := Native.launch_exact m n b l t h id before binding
  have copy := launch_correct m n b l t h before
  refine ⟨⟨none,cacheObject,m .cache,launch m (config m n b) l t before⟩,
    ?_,rfl,rfl,rfl,copy,?_⟩
  · simp [runWrapper,(checkInputs_iff m n b l t).mpr h,execution]
  · exact byte_frame m (config m n b) l t before _ copy

#axiomsClean correct
end Direct
#axiomsClean store_bytes_disjoint
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
