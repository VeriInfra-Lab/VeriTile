import bench.tritonbench_g.chunk_linear_attn.WholeReal
import bench.tritonbench_g.chunk_delta_fwd.Proofs.Grid
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Physical
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

structure Buffer where
  region : RegionName
  base : Nat
  words : Nat
  deriving DecidableEq

def Separate (a b : Buffer) : Prop := a.base+a.words≤b.base ∨ b.base+b.words≤a.base
def Contains (b : Buffer) (address : Nat) : Prop := b.base≤address ∧ address<b.base+b.words
def write {α : Type} (b : Buffer) (values : RegionName → Nat → α) (memory : Nat → α) : Nat → α :=
  fun address => if b.base≤address ∧ address<b.base+b.words then values b.region (address-b.base) else memory address
def publish {α : Type} (buffers : List Buffer) (values : RegionName → Nat → α) (memory : Nat → α) : Nat → α :=
  buffers.foldl (fun m b => write b values m) memory

theorem write_read {α : Type} (b : Buffer) (values : RegionName → Nat → α) (memory : Nat → α)
    (o : Nat) (ho : o<b.words) : write b values memory (b.base+o)=values b.region o := by
  simp [write,ho]
theorem write_frame {α : Type} (b : Buffer) (values : RegionName → Nat → α) (memory : Nat → α)
    (address : Nat) (h : ¬Contains b address) : write b values memory address=memory address := by
  unfold Contains at h
  simp only [write,if_neg h]
theorem separate_miss (a b : Buffer) (h : Separate a b) (o : Nat) (ho : o<a.words) :
    ¬Contains b (a.base+o) := by
  unfold Separate at h
  unfold Contains
  rcases h with h|h <;> omega

theorem publish_read {α : Type} (buffers : List Buffer) (values : RegionName → Nat → α) (memory : Nat → α)
    (b : Buffer) (hb : b∈buffers) (o : Nat) (ho : o<b.words)
    (separate : ∀ x∈buffers, x≠b → Separate b x) :
    publish buffers values memory (b.base+o)=values b.region o := by
  apply VeriTile.Bench.Repairs.ChunkDeltaFwd.list_read
    _ (fun m : Nat → α => m (b.base+o)) buffers b memory _ hb
  · intro m
    exact write_read b values m o ho
  · intro other hother hn m
    exact write_frame other values m _ (separate_miss b other (separate other hother hn) o ho)

theorem publish_frame {α : Type} (buffers : List Buffer) (values : RegionName → Nat → α) (memory : Nat → α)
    (address : Nat) (h : ∀ b∈buffers, ¬Contains b address) :
    publish buffers values memory address=memory address := by
  apply list_observe _ (fun m : Nat → α => m address)
  intro b hb m
  exact write_frame b values m address (h b hb)

theorem publish_input {α : Type} (buffers : List Buffer) (values : RegionName → Nat → α) (memory : Nat → α)
    (input : Buffer) (separate : ∀ b∈buffers, Separate input b) (o : Nat) (ho : o< input.words) :
    publish buffers values memory (input.base+o)=memory (input.base+o) := by
  apply publish_frame
  intro b hb
  exact separate_miss input b (separate b hb) o ho

def outputRegions (seed final : Bool) : List RegionName :=
  ["history","output","historyGradient","queryGradient","keyGradient","valuePartials","valueGradient"]++
    (if final then ["final"] else [])++(if seed then ["initialGradient"] else [])
def extent (p : Parameters) (r : RegionName) : Nat :=
  if r="history" ∨ r="historyGradient" then p.B*p.H*p.NT*p.K*p.V
  else if r="queryGradient" ∨ r="keyGradient" then p.B*p.H*p.T*p.K
  else if r="valuePartials" then keyBlocks p*(p.B*p.H)*p.T*p.V
  else if r="final" ∨ r="initialGradient" then p.B*p.H*p.K*p.V
  else p.B*p.H*p.T*p.V

/-- Allocation metadata only: exact output spans, distinct fresh buffers and
separation from the caller's inputs. No numerical payload is assumed. -/
structure Layout (p : Parameters) (seed final : Bool) where
  outputs : List Buffer
  inputs : List Buffer
  regions : outputs.map Buffer.region=outputRegions seed final
  sizes : ∀ b∈outputs, b.words=extent p b.region
  distinct : ∀ a∈outputs, ∀ b∈outputs, a.region=b.region → a=b
  fresh : ∀ a∈outputs, ∀ b∈outputs, a≠b → Separate a b
  inputFresh : ∀ a∈inputs, ∀ b∈outputs, Separate a b

structure Post {α : Type} (p : Parameters) (seed final : Bool) (layout : Layout p seed final)
    (values : RegionName → Nat → α) (before after : Nat → α) : Prop where
  outputs : ∀ b∈layout.outputs, ∀ o, o<b.words → after (b.base+o)=values b.region o
  inputs : ∀ b∈layout.inputs, ∀ o, o<b.words → after (b.base+o)=before (b.base+o)
  frame : ∀ address, (∀ b∈layout.outputs, ¬Contains b address) → after address=before address

theorem publication_complete {α : Type} (p : Parameters) (seed final : Bool) (layout : Layout p seed final)
    (values : RegionName → Nat → α) (before : Nat → α) :
    Post p seed final layout values before (publish layout.outputs values before) := by
  constructor
  · intro b hb o ho
    exact publish_read layout.outputs values before b hb o ho
      (fun x hx hn => layout.fresh b hb x hx (Ne.symm hn))
  · intro b hb o ho
    exact publish_input layout.outputs values before b (layout.inputFresh b hb) o ho
  · exact publish_frame layout.outputs values before

/-- Word addresses are translated to the four-byte FP32 pointer ABI. -/
theorem byte_address_injective (a b : Nat) : 4*a=4*b ↔ a=b := by omega

#axiomsClean publish_read
#axiomsClean publish_frame
#axiomsClean publish_input
#axiomsClean publication_complete
#axiomsClean byte_address_injective
#specNonCircular Layout avoiding [publish,Post]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Physical
