import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Storage
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Host
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.DirectHost
open VeriTile.Triton VeriTile.Torch Layout
noncomputable section
abbrev Heap (β : Type) := Nat → β
structure Allocated (β : Type) where
  identity : Nat
  tensor : TensorMetadata
  heap : Heap β

structure API (β : Type) where
  launchDevice : Device
  emptyLike : Heap β → Except String (Allocated β)
  setup : Nat → Config → Except String Unit

/-- The only effects are one output allocation and actual source stores. The
lift/commit adapters interpret native addresses; they are not Tensor copies. -/
def launches {c : Shape} {S β : Type} (src : Host.Source c S β) (api : API β) :
    Nat → List Config → S → Except String S
  | _,[],s => .ok s
  | n,cfg::rest,s => match api.setup n cfg with
    | .error e => .error e
    | .ok () => match src.launch cfg s with
      | none => .error "source execution failed"
      | some t => launches src api (n+1) rest t

def Setups {β : Type} (api : API β) : Nat → List Config → Prop
  | _,[] => True
  | n,cfg::rest => api.setup n cfg=.ok () ∧ Setups api (n+1) rest

def run {S β : Type} (c : Shape) (p : Storage.Placement c)
    (src : Host.Source c S β) (entry : Heap β → S) (memory : S → RegionName → Nat → β)
    (api : API β) (admitted : Bool) (trials : List Config) (selected : Config) (caller : Heap β) :
    Except String (Allocated β) :=
  if admitted then match api.emptyLike caller with
    | .error e => .error e
    | .ok a => if c.B*c.H*c.T=0 then .ok a else
      match launches src api 0 (trials++[selected]) (entry a.heap) with
      | .error e => .error e
      | .ok t => .ok {a with heap:=Storage.commit c p a.heap (memory t)}
  else .error "unsupported input"

theorem launches_of_setups {c : Shape} {S β : Type} (src : Host.Source c S β) (api : API β)
    (n : Nat) (cs : List Config) (setups : Setups api n cs) (s t : S)
    (he : Host.sourceLaunches src cs s=some t) : launches src api n cs s=.ok t := by
  induction cs generalizing n s with
  | nil => cases he; rfl
  | cons cfg rest ih =>
    rcases setups with ⟨setup,setups⟩
    simp only [Host.sourceLaunches] at he
    cases runeq : src.launch cfg s with
    | none => simp [runeq] at he
    | some mid =>
      simp only [runeq,Option.bind_some] at he
      simp only [launches,setup,runeq]
      exact ih (n+1) setups mid he

theorem successful_return {S β : Type} (c : Shape) (p : Storage.Placement c)
    (src : Host.Source c S β) (entry : Heap β → S) (memory : S → RegionName → Nat → β)
    (api : API β) (trials : List Config) (selected : Config) (caller : Heap β)
    (a : Allocated β) (allocated : api.emptyLike caller=.ok a) (nonempty : c.B*c.H*c.T≠0)
    (t : S) (he : Host.sourceLaunches src (trials++[selected]) (entry a.heap)=some t)
    (setups : Setups api 0 (trials++[selected])) :
    run c p src entry memory api Bool.true trials selected caller=
      .ok {a with heap:=Storage.commit c p a.heap (memory t)} := by
  simp only [run,allocated,nonempty,if_false,if_true]
  rw [launches_of_setups src api 0 _ setups _ _ he]

def metadata (s : TensorMetadata) (c : Shape) (p : Storage.Placement c) {β : Type} (a : Allocated β) : Prop :=
  a.tensor.shape=[c.B,c.H,c.T] ∧ a.tensor.dtype=.floating .fp32 ∧ a.tensor.layout=.strided ∧
    a.tensor.isContiguous=Bool.true ∧ a.tensor.device=s.device ∧
    (c.B*c.H*c.T≠0 → a.tensor.dataPtr.toNat=p.backing.output.base)

def allocationWrites (c : Shape) (p : Storage.Placement c) (addr : Nat) : Prop :=
  p.backing.output.base≤addr ∧ addr<p.backing.output.base+p.backing.output.cells*4

theorem writes_within_allocation (c : Shape) (p : Storage.Placement c) (addr : Nat)
    (written : Storage.writes c p addr) : allocationWrites c p addr := by
  obtain ⟨row,hr,time,ht,eq⟩ := written
  have inside := (Storage.offset_bound c row time hr ht).trans_le p.backing.outputCells
  rw [eq]
  unfold allocationWrites Storage.address
  omega

/-- Ordinary direct input and fresh allocator semantics, with no source or
prefix result premise. Capacities are remaining windows from data_ptr. -/
structure AllocationLaws {β : Type} (s : TensorMetadata) (c : Shape) (p : Storage.Placement c)
    (caller : Heap β) (a : Allocated β) (live : Nat → Prop) : Prop where
  metadata : metadata s c p a
  input : ∀ off, off<p.backing.input.cells → a.heap (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)
  frame : ∀ addr, ¬allocationWrites c p addr → a.heap addr=caller addr
  liveFrame : ∀ addr, live addr → a.heap addr=caller addr
  fresh : ∀ addr, live addr → ¬Storage.writes c p addr

#axiomsClean successful_return
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.DirectHost
