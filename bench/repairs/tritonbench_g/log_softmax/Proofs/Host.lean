import bench.repairs.tritonbench_g.log_softmax.Proofs.Contract
import bench.repairs.tritonbench_g.log_softmax.Proofs.Congruence

namespace VeriTile.Bench.Repairs.LogSoftmax.Host
open VeriTile.Torch Address Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open Classical

/-- The dense views produced by the original contiguous calls and a fresh
output allocation. Input copies are Torch library operations, not extra
snapshots in the candidate. Only the output must be separate from inputs. -/
structure Buffers (M N K : Nat) where
  first : Dense (M*N*K)
  second : Dense (M*N*K)
  output : Dense (M*N*K)
  firstApart : Disjoint output first
  secondApart : Disjoint output second

def Buffers.launch (b : Buffers M N K) (ops : Ops A) (direction : Direction)
    (selected : Grid.Selected M N K) : Launch A M N K selected.C :=
  ⟨direction,ops,selected.plan,b.first,b.second,b.output,b.firstApart,b.secondApart⟩

def trials (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K) :
    List (Grid.Selected M N K) → Store → Store
  | [],s => s
  | selected::rest,s => trials api ops direction b rest
      (execute api (b.launch ops direction selected) selected.order s)

theorem trials_inputs (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K)
    (probes : List (Grid.Selected M N K)) (s : Store) :
    (∀ q j,rowRead api b.first (trials api ops direction b probes s) q j=rowRead api b.first s q j) ∧
    (∀ q j,rowRead api b.second (trials api ops direction b probes s) q j=rowRead api b.second s q j) := by
  induction probes generalizing s with
  | nil => exact ⟨by intros; rfl,by intros; rfl⟩
  | cons p rest ih =>
    have one := execute_inputs api (b.launch ops direction p) p.order s
    have tail := ih (execute api (b.launch ops direction p) p.order s)
    exact ⟨fun q j => (tail.1 q j).trans (one.1 q j),fun q j => (tail.2 q j).trans (one.2 q j)⟩

theorem trials_frame (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K)
    (probes : List (Grid.Selected M N K)) (s : Store) (i : Nat) (outside : ¬b.output.contains i) :
    api.byte (trials api ops direction b probes s) i=api.byte s i := by
  induction probes generalizing s with
  | nil => rfl
  | cons p rest ih =>
    exact (ih _).trans (execute_frame api (b.launch ops direction p) p.order s i outside)

def dispatch (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K)
    (probes : List (Grid.Selected M N K)) (selected : Grid.Selected M N K) (s : Store) : Store :=
  if M*N*K=0 then s else execute api (b.launch ops direction selected) selected.order
    (trials api ops direction b probes s)

/-- Every returned cell consumes the reached input buffers. Autotuning probes
can change output bytes, but cannot change either input. Final coverage is
derived from the exact two-dimensional grid and every retained BM choice. -/
theorem dispatch_values (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K)
    (probes : List (Grid.Selected M N K)) (selected : Grid.Selected M N K) (s : Store) (q : Coord M N K) :
    api.read (dispatch api ops direction b probes selected s) (b.output.address (linear q))=
      value direction ops selected.plan (rowRead api b.first s q) (rowRead api b.second s q) q.2.1.val := by
  have nonempty : M*N*K≠0 := by
    have hm : 0<M := Nat.zero_lt_of_lt q.1.isLt
    have hn : 0<N := Nat.zero_lt_of_lt q.2.1.isLt
    have hk : 0<K := Nat.zero_lt_of_lt q.2.2.isLt
    positivity
  rw [dispatch,if_neg nonempty]
  exact (execute_readback api (b.launch ops direction selected) selected.order _ q
    (Grid.selected_complete selected q)).trans
    (scalar_stable api (b.launch ops direction selected) s _ (trials_inputs api ops direction b probes s) q)

theorem dispatch_frame (api : API A Store Byte) (ops : Ops A) (direction : Direction) (b : Buffers M N K)
    (probes : List (Grid.Selected M N K)) (selected : Grid.Selected M N K) (s : Store)
    (i : Nat) (outside : ¬b.output.contains i) :
    api.byte (dispatch api ops direction b probes selected s) i=api.byte s i := by
  unfold dispatch
  split
  · rfl
  · exact (execute_frame api (b.launch ops direction selected) selected.order _ i outside).trans
      (trials_frame api ops direction b probes s i outside)

structure Context (a : Arguments) where
  saved : Dense (a.M*a.N*a.K)
  dim : Nat

structure ForwardResult (a : Arguments) (Store : Type) where
  memory : Store
  shape : List Nat
  dtype : DType
  device : Device
  output : Dense (a.M*a.N*a.K)
  context : Context a

def forwardCall (api : API A Store Byte) (ops : Ops A) (a : Arguments)
    (b : Buffers a.M a.N a.K) (probes : List (Grid.Selected a.M a.N a.K))
    (selected : Grid.Selected a.M a.N a.K) (s : Store) : Except InputError (ForwardResult a Store) := do
  let _ ← checkInputs a
  pure ⟨dispatch api ops .forward b probes selected s,a.input.shape,.floating .fp32,a.input.device,
    b.output,⟨b.output,a.axis⟩⟩

theorem forward_success (api : API A Store Byte) (ops : Ops A) (a : Arguments) (h : InputContract a)
    (b : Buffers a.M a.N a.K) (probes : List (Grid.Selected a.M a.N a.K))
    (selected : Grid.Selected a.M a.N a.K) (s : Store) :
    forwardCall api ops a b probes selected s=.ok
      ⟨dispatch api ops .forward b probes selected s,a.input.shape,.floating .fp32,a.input.device,
        b.output,⟨b.output,a.axis⟩⟩ := by
  simp [forwardCall,(checkInputs_iff a).mpr h]

structure BackwardResult (a : Arguments) (Store : Type) where
  memory : Store
  shape : List Nat
  dtype : DType
  device : Device
  output : Dense (a.M*a.N*a.K)
  dimGradient : Option Unit
  dtypeGradient : Option Unit

/-- The first input descriptor is the saved forward output. Autograd's
save/version checks and contiguous materialization are ordinary library
boundaries; no detached or cloned saved tensor is introduced. -/
def backwardCall (api : API A Store Byte) (ops : Ops A) (a : BackwardArguments)
    (ctx : Context a.saved) (b : Buffers a.saved.M a.saved.N a.saved.K)
    (saved : ctx.saved=b.first) (dimension : ctx.dim=a.saved.axis)
    (probes : List (Grid.Selected a.saved.M a.saved.N a.saved.K))
    (selected : Grid.Selected a.saved.M a.saved.N a.saved.K) (s : Store) :
    Except InputError (BackwardResult a.saved Store) := do
  let _ ← checkBackward a
  let savedBuffers : Buffers a.saved.M a.saved.N a.saved.K :=
    {b with first := ctx.saved, firstApart := by simpa only [saved] using b.firstApart}
  pure ⟨dispatch api ops .backward savedBuffers probes selected s,a.saved.input.shape,.floating .fp32,
      a.saved.input.device,b.output,none,none⟩

theorem backward_success (api : API A Store Byte) (ops : Ops A) (a : BackwardArguments) (h : BackwardContract a)
    (ctx : Context a.saved) (b : Buffers a.saved.M a.saved.N a.saved.K)
    (saved : ctx.saved=b.first) (dimension : ctx.dim=a.saved.axis)
    (probes : List (Grid.Selected a.saved.M a.saved.N a.saved.K))
    (selected : Grid.Selected a.saved.M a.saved.N a.saved.K) (s : Store) :
    backwardCall api ops a ctx b saved dimension probes selected s=.ok
      ⟨dispatch api ops .backward b probes selected s,a.saved.input.shape,.floating .fp32,
        a.saved.input.device,b.output,none,none⟩ := by
  simp [backwardCall,(checkBackward_iff a).mpr h,saved]

#axiomsClean dispatch_values
#axiomsClean dispatch_frame
#axiomsClean forward_success
#axiomsClean backward_success
end
end VeriTile.Bench.Repairs.LogSoftmax.Host
