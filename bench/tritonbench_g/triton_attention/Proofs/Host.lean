import bench.tritonbench_g.triton_attention.Proofs.NativeHostOrder
import bench.tritonbench_g.triton_attention.Proofs.TritonAttention_Grid
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Host
open Native
noncomputable section

inductive DType where | fp32 | fp16 | other deriving DecidableEq
structure Tensor where
  buffer : Buffer
  shape : List Nat
  strides : List Nat
  device : Nat
  dtype : DType
  isTensor : Bool
  strided : Bool
  cuda : Bool
  resolved : Bool
  deriving DecidableEq

def dimensions (c : Shape) := [c.batches,c.heads,c.rows,c.dim]
def strides (c : Shape) := [c.heads*c.rows*c.dim,c.rows*c.dim,c.dim,1]

def tensor (c : Shape) (device : Nat) (buffer : Buffer) : Tensor :=
  ⟨buffer,dimensions c,strides c,device,.fp32,true,true,true,true⟩
def rowTensor (c : Shape) (device : Nat) (buffer : Buffer) : Tensor :=
  ⟨buffer,[c.groups,c.rows],[c.rows,1],device,.fp32,true,true,true,true⟩

structure Arguments where
  q : Tensor
  k : Tensor
  v : Tensor
  /-- The actual FP32 launch word after the validated Python float conversion. -/
  scale : Word

def InputChecks (c : Shape) (a : Arguments) : Prop :=
  a.q = tensor c a.q.device .q ∧ a.k = tensor c a.q.device .k ∧
  a.v = tensor c a.q.device .v ∧ 0 < a.scale.toNat ∧ a.scale.toNat < 0x7f800000

def GradientChecks (c : Shape) (a : Arguments) (g : Tensor) : Prop :=
  g.isTensor = true ∧ g.strided = true ∧ g.shape = dimensions c ∧
  g.device = a.q.device ∧ g.dtype = .fp32 ∧ g.cuda = true ∧ g.resolved = true

inductive Stage where | forward | preprocess | backward deriving DecidableEq
structure Launch where
  stage : Stage
  gridX : Nat
  gridY : Nat
  blockM : Nat
  blockN : Nat
  dim : Nat
  warps : Nat
  stages : Nat
  deriving DecidableEq

def forwardLaunch (c : Shape) : Launch :=
  ⟨.forward,c.blocks,c.groups,128,128,c.dim,if c.dim ≤ 64 then 4 else 8,2⟩
def preprocessLaunch (c : Shape) : Launch :=
  ⟨.preprocess,c.blocks*c.groups,1,128,128,c.dim,4,3⟩
def backwardLaunch (c : Shape) : Launch :=
  ⟨.backward,c.groups,1,128,128,c.dim,8,1⟩

structure Runtime where
  launch : Launch → Word → Memory Word → Option (Memory Word)
  zerosDQ : Memory Word → Option (Memory Word)
  contiguousGradient : Tensor → Memory Word → Option (Memory Word)

/-- Ordinary tensor/compiler/runtime assumptions: the pinned source is
materialized with its exact argument, loop, memory, constant and instruction
semantics. Numerical functions here remain the actual ISA; no mathematical
attention, gradient, reference-profile or final-output equality appears.
Memory is the ordinary decoded logical view after fresh allocation. The
incoming G view already denotes do.contiguous()'s logical values. -/
structure Binding (c : Shape) (actual : Instructions) (owners : Owners)
    (runtime : Runtime) (g : Tensor) : Prop where
  forward : ∀ scale s, runtime.launch (forwardLaunch c) scale s =
    some (Native.forward c (ops actual) owners s scale)
  preprocess : ∀ scale s, runtime.launch (preprocessLaunch c) scale s =
    some (Native.preprocess c (ops actual) owners s)
  backward : ∀ scale s, runtime.launch (backwardLaunch c) scale s =
    some (backwardFrom c (ops actual) scale s c.blocks)
  zeros : ∀ s, runtime.zerosDQ s = some (zeroDQ c (ops actual) s)
  contiguous : ∀ s, runtime.contiguousGradient g s = some s

structure Context where
  q : Tensor
  k : Tensor
  v : Tensor
  o : Tensor
  l : Tensor
  m : Tensor
  scale : Word
  grid : Launch

structure Result where
  output : Tensor
  context : Context
  dq : Tensor
  dk : Tensor
  dv : Tensor
  scaleGradient : Option Tensor
  memory : Memory Word

def result (c : Shape) (a : Arguments) (s : Memory Word) : Result :=
  { output := tensor c a.q.device .o
    context := ⟨a.q,a.k,a.v,tensor c a.q.device .o,rowTensor c a.q.device .l,
      rowTensor c a.q.device .m,a.scale,forwardLaunch c⟩
    dq := tensor c a.q.device .dq
    dk := tensor c a.q.device .dk
    dv := tensor c a.q.device .dv
    scaleGradient := none
    memory := s }

/-- Complete forward and backward host invocation, preserving the source
order and the final (dq,dk,dv,None) return contract. Noncontiguous incoming
gradients are admitted through the ordinary contiguous-copy contract. -/
def invoke (c : Shape) (a : Arguments) (g : Tensor) (runtime : Runtime)
    (s : Memory Word) : Option Result := by
  classical
  exact if InputChecks c a then do
    let f ← runtime.launch (forwardLaunch c) a.scale s
    if ¬ GradientChecks c a g then none else do
    let gc ← runtime.contiguousGradient g f
    let z ← runtime.zerosDQ gc
    let pre ← runtime.launch (preprocessLaunch c) a.scale z
    let final ← runtime.launch (backwardLaunch c) a.scale pre
    pure (result c a final)
  else none

theorem rejected (c : Shape) (a : Arguments) (g : Tensor) (runtime : Runtime)
    (s : Memory Word) (h : ¬(InputChecks c a ∧ GradientChecks c a g)) :
    invoke c a g runtime s = none := by
  by_cases hi : InputChecks c a
  · have hg : ¬ GradientChecks c a g := fun hg => h ⟨hi,hg⟩
    simp only [invoke,if_pos hi]
    cases runtime.launch (forwardLaunch c) a.scale s <;> simp [hg]
  · simp [invoke,hi]

theorem success (c : Shape) (a : Arguments) (g : Tensor) (runtime : Runtime)
    (s : Memory Word) (actual : Instructions) (owners : Owners)
    (checks : InputChecks c a ∧ GradientChecks c a g) (binding : Binding c actual owners runtime g) :
    invoke c a g runtime s = some (result c a (Native.run c (ops actual) owners s a.scale)) := by
  simp only [invoke,if_pos checks.1,not_true_eq_false,checks.2,if_false,binding.forward,binding.contiguous,binding.zeros,
    binding.preprocess,binding.backward,Option.pure_def]
  rw [← hostOrderRun_eq_run]
  rfl

/-- Public full word-graph certificate. Compiler/Tensor/runtime contracts
are ordinary assumptions; exactly eleven local ISA conformance families are
the separately registered numerical boundary. -/
theorem whole_host (c : Shape) (a : Arguments) (g : Tensor) (runtime : Runtime)
    (s : Memory Word) (actual reference : Instructions) (owners : Owners)
    (checks : InputChecks c a ∧ GradientChecks c a g) (binding : Binding c actual owners runtime g)
    (numeric : LocalConformance actual reference) :
    ∃ r, invoke c a g runtime s = some r ∧
      r.output = tensor c a.q.device .o ∧
      r.dq = tensor c a.q.device .dq ∧ r.dk = tensor c a.q.device .dk ∧
      r.dv = tensor c a.q.device .dv ∧ r.scaleGradient = none ∧
      Native.Correct c reference owners s r.memory a.scale ∧
      (∀ i : Coord c, offset c i < 2147483647 ∧
        4 * offset c i < 9223372036854775807 ∧
        4 * (c.elements + 16384) < 9223372036854775807) := by
  refine ⟨result c a (Native.run c (ops actual) owners s a.scale),
    success c a g runtime s actual owners checks binding,rfl,rfl,rfl,rfl,rfl,?_,?_⟩
  · exact whole_native c actual reference owners s a.scale numeric
  · exact machine_offsets c

#axiomsClean rejected
#axiomsClean success
#axiomsClean whole_host
#specNonCircular Binding avoiding [Native.Correct,Native.whole_native,whole_host]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Host
