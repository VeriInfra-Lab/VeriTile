import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostAPI

namespace VeriTile.Bench.Repairs.ParallelAttention.HostAPI
noncomputable section
open scoped BigOperators

/-- The O/Z handles retained by the ordinary division graph still contain the
actual forward values, including when the public result uses a new buffer. -/
def RawSaved (s : Shape) (o z : Tensor) (atForward later : Memory α) : Prop :=
  (∀ bh t e,bh<s.B*s.H → t<s.T → e<s.V → o.read later bh t e=o.read atForward bh t e) ∧
  (∀ bh t,bh<s.B*s.H → t<s.T → z.read later bh t 0=z.read atForward bh t 0)

/-- Explicit Real interpretation of the ordinary Tensor division VJP.
Every factor comes from the current incoming tensor or a retained O/Z handle.
The value-axis sum is finite code, not an assumed reduction result. -/
def propagationReads (s : Shape) (o z : Tensor) (dy : View) : Set Nat :=
  {address | ∃ bh t e,bh<s.B*s.H ∧ t<s.T ∧ e<s.V ∧
    address=o.base+dense o.metadata.time o.metadata.features bh t e} ∪
  {address | ∃ bh t,bh<s.B*s.H ∧ t<s.T ∧
    address=z.base+dense z.metadata.time z.metadata.features bh t 0} ∪
  {address | ∃ bh t e,bh<s.B*s.H ∧ t<s.T ∧ e<s.V ∧ address=dy.address s bh t e}

def PropagationFresh (s : Shape) (o z : Tensor) (dy : View) (dest : OutputKind → Nat) : Prop :=
  ∀ ev∈events s false forwardKinds,tensorAddress dest ev∉propagationReads s o z dy

/-- Ordinary Tensor graph evaluation followed by materialization into the
actual contiguous do/dz allocations. There is no gradient-output callback. -/
def PropagationLayout (s : Shape) (dest : OutputKind → Nat) : Prop :=
  ∀ e∈events s false forwardKinds,∀ f∈events s false forwardKinds,
    tensorAddress dest e=tensorAddress dest f → e=f


end
end VeriTile.Bench.Repairs.ParallelAttention.HostAPI
