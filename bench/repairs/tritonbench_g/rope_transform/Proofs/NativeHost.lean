import bench.repairs.tritonbench_g.rope_transform.Proofs.Native
import bench.repairs.tritonbench_g.rope_transform.Proofs.Host
namespace VeriTile.Bench.Repairs.RopeTransform.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def entry (memory : RegionName → Nat → Word) : State Term where
  mem := fun name o => Cell.mk .fp32 (.leaf name o (memory name o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d
def observe (I : Instructions) (S : Schedule) (s : State Term) (name : RegionName) (o : Nat) :=
  eval I S ((s.mem name o).read .fp32)
def expectedWord (I : Instructions) (S : Schedule) (r : Request)
    (memory : RegionName → Nat → Word) (input : Host.Tensor) (key : Bool) (b h s c : Nat) :=
  eval I S (Host.expected algebra r (entry memory) input key b h s c)

/-- Full host output words from actual successful source execution. Local
instruction semantics and same-compilation schedule are parameters; the input
binding contains only caller words, never an expected whole output. -/
theorem whole_native (I : Instructions) (S : Schedule) (r : Request)
    (memory : RegionName → Nat → Word) (call : Host.Preparation Term r)
    (before : call.before=entry memory) :
    ∃ result,Host.invoke algebra r call=some result ∧
      (∀ key,result.tensor key=call.output key) ∧
      result.cosine=call.cosine ∧ result.sine=call.sine ∧
      (∀ key,(result.tensor key).shape=Host.shape r key ∧ (result.tensor key).dtype="float32" ∧
        (result.tensor key).device=(call.input Bool.false).device) ∧
      (∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
        observe I S result.state (result.tensor key).region ((result.tensor key).offset b h s c)=
          expectedWord I S r memory (call.input key) key b h s c) ∧
      (∀ name o,Host.Outside r name o → observe I S result.state name o=memory name o) ∧
      (∀ o,observe I S result.state "C" o=memory "C" o ∧ observe I S result.state "S" o=memory "S" o) := by
  obtain ⟨result,he,ht,hc,hs,hmeta,hv,hf,hcoeff⟩ := Host.whole_typed algebra r call
  refine ⟨result,he,ht,hc,hs,hmeta,?_,?_,?_⟩
  · intro key b h s c hb hh hs hc
    unfold observe
    rw [hv key b h s c hb hh hs hc,before]
    rfl
  · intro name o ho
    unfold observe
    rw [hf name o ho,before]
    rfl
  · intro o
    constructor
    · unfold observe; rw [(hcoeff o).1,before]; rfl
    · unfold observe; rw [(hcoeff o).2,before]; rfl

#axiomsClean whole_native
#specNonCircular entry avoiding [FP.Structural.exec,Host.invoke,Grid.run]
#specNonCircular expectedWord avoiding [FP.Structural.exec,Host.invoke,Grid.run,value,T.laneResult]
end
end VeriTile.Bench.Repairs.RopeTransform.Native
