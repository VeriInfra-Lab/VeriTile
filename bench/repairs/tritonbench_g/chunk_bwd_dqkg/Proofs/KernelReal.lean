import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelTensor
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelObjective

/-! Complete actual-quote execution under the precision-erased Real
interpretation, independent local VJP formulas, every output Tensor cell,
and an exact backing-byte frame. The output buffers already exist. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelReal
open VeriTile.Triton Whole KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
def memory (b : Buffers c) (read : Nat → ℝ) (r : RegionName) (i : Nat) : ℝ :=
  if i<(b.tensor r).metadata.numel then read ((b.view r).address i) else 0
def initial (b : Buffers c) (read : Nat → ℝ) : BlockState where
  mem:=fun r i=>.real (memory b read r i)
  regs:=fun _ _ _=>none
  pids:=fun _=>0
def image (s : BlockState) : FP.Structural.State ℝ where
  mem:=fun r i=>.mk .real (s.readMem r i)
  regs:=fun _ _ _=>none
  pids:=s.pids
  numPids:=s.numPids
  undef:=fun d _ _=>FP.Structural.defaultValue d
def observe (_ : RegionName) (_ : Nat) (x : FP.Structural.Cell ℝ) : ℝ := x.read .real
theorem input_address (b : Buffers c) (read : Nat → ℝ) (r : RegionName) (i : Nat)
    (hi : i<(b.tensor r).metadata.numel) :
    (initial b read).readMem r i=read ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)) := by
  simp [initial,BlockState.readMem,memory,hi,logical_address b r i hi]

structure Result (c : Config) (before final : BlockState) : Prop where
  q : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
    final.readMem "dq" ((h.val*c.T+t.val)*c.K+e.val)=Spec.dq c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val
  k : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),
    final.readMem "dk" ((h.val*c.T+t.val)*c.K+e.val)=Spec.dk c (Spec.views c before.readMem) h.val (t.val/64) (t.val%64) e.val
  g : ∀ (key : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),
    final.readMem "dg" ((key.val*c.heads+h.val)*c.T+t.val)=Spec.dg c (Spec.views c before.readMem) key.val h.val (t.val/64) (t.val%64)
  frame : ∀ r i,¬Writes c r i → final.mem r i=before.mem r i
theorem source (c : Config) (s : BlockState) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final,Whole.runGrid c order s=some final ∧ Result c s final := by
  obtain ⟨final,he,hq,hk,hg,hf⟩ := Whole.whole_result c s order ho
  exact ⟨final,he,⟨fun h t e=>(hq h t e).trans (Spec.logical_q c s h t e),
    fun h t e=>(hk h t e).trans (Spec.logical_k c s h t e),
    fun key h t=>(hg key h t).trans (Spec.logical_g c s key h t),fun r i hn=>(hf r i hn).symm⟩⟩

theorem complete {Store Byte : Type} (api : Memory.API ℝ Store Byte)
    (b : Buffers c) (store : Store) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final,Whole.runGrid c order (initial b (api.read store))=some final ∧
      Result c (initial b (api.read store)) final ∧
      Shared.TensorImage api (placement b) observe (image final) store := by
  obtain ⟨final,he,hr⟩ := source c (initial b (api.read store)) order ho
  exact ⟨final,he,hr,Shared.complete api (placement b) observe (image final) store⟩

theorem tensor_value {Store Byte : Type} (api : Memory.API ℝ Store Byte) (b : Buffers c)
    (store : Store) (final : BlockState) (r : RegionName) (i : Nat) (active : Writes c r i) :
    api.read (Shared.commit api (placement b) observe (image final) store)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))=final.readMem r i :=
  Shared.commit_output api (placement b) observe (image final) store r i active

#axiomsClean input_address
#axiomsClean source
#axiomsClean complete
#axiomsClean tensor_value
#specNonCircular Result avoiding [Whole.runGrid,VeriTile.Triton.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelReal
