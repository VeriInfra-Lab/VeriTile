import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Machine
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Grid
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.FPEquiv

/- The source launcher selects q.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
namespace Host

structure Tensor where
  shape : List Nat
  device : Nat
  cuda : Bool
  fp32 : Bool
  contiguous : Bool
  conjugate : Bool
  negative : Bool
  base : Nat

structure Arguments where
  q : Tensor
  k : Tensor
  v : Tensor
  store : Option Bool
  ifcond : Option Bool

def mainShape (c : Shape) : List Nat := [c.B,c.H,c.T,c.D]
def historyShape (c : Shape) : List Nat := [c.B,c.H,c.chunks*c.D,c.D]

/-- Metadata checks after dimension validation. The Shape constructor records
the source's power-of-two D>=16 validation, with no experimental D cap. -/
def InputChecks (c : Shape) (args : Arguments) : Prop :=
  ∀ t∈[args.q,args.k,args.v], t.shape=mainShape c ∧ t.device=args.q.device ∧
    t.cuda=Bool.true ∧ t.fp32=Bool.true ∧ t.contiguous=Bool.true ∧
    t.conjugate=Bool.false ∧ t.negative=Bool.false

def Empty (c : Shape) : Prop := c.T=0 ∨ c.heads=0
instance (c : Shape) : Decidable (Empty c) := inferInstanceAs (Decidable (c.T=0 ∨ c.heads=0))

structure Plan where
  grid : Nat
  bt : Nat
  bd : Nat
  nt : Nat
  qHead : Nat
  qt : Nat
  qd : Nat
  hHead : Nat
  ht : Nat
  hd : Nat
  warps : Nat
  stages : Nat
  store : Bool
  ifcond : Bool
  deriving DecidableEq

def plan (c : Shape) (store ifcond : Bool) : Plan :=
  ⟨c.heads,32,c.D,c.chunks,c.headSize,c.D,1,c.historyHead,c.D,1,
    4,if c.D≤64 then 3 else 2,store,ifcond⟩

/-- Successful ordinary allocation: fresh dense O/H, preserved input bases,
and complete FP32-word backing extents. Q/K/V are allowed to alias. -/
structure AllocationReceipt (c : Shape) (args : Arguments) where
  abi : ABI c
  qBase : abi.q.base=args.q.base
  kBase : abi.k.base=args.k.base
  vBase : abi.v.base=args.v.base

structure Result where
  output : Tensor
  history : Tensor
  memory : Memory (BitVec 32)
  launched : Bool

def outputTensor (c : Shape) (args : Arguments) (abi : ABI c) : Tensor :=
  ⟨mainShape c,args.q.device,Bool.true,Bool.true,Bool.true,Bool.false,Bool.false,abi.o.base⟩
def historyTensor (c : Shape) (args : Arguments) (abi : ABI c) : Tensor :=
  ⟨historyShape c,args.q.device,Bool.true,Bool.true,Bool.true,Bool.false,Bool.false,abi.h.base⟩
def result (c : Shape) (args : Arguments) (abi : ABI c) (m : Memory (BitVec 32)) (launched : Bool) : Result :=
  ⟨outputTensor c args abi,historyTensor c args abi,m,launched⟩

/-- Exact public return convention: forward returns only O. The complete
internal result retains the H side effect and full word memory for framing. -/
def publicReturn (r : Result) : Tensor := r.output

/-- A typed successful-shape wrapper model. None includes rejected metadata,
non-bool flags, allocation failure, and machine-guard failure. The ordinary
Torch/JIT boundary binds this to the unchanged Python call convention. -/
def run (c : Shape) (args : Arguments) (allocation : Option (AllocationReceipt c args))
    (backend : AllocationReceipt c args → Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32)) : Option Result := by
  classical
  exact if InputChecks c args then
    match args.store,args.ifcond,allocation with
    | some store,some ifcond,some receipt =>
      if Empty c then some (result c args receipt.abi initial Bool.false)
      else if Machine.Domain c receipt.abi then
        some (result c args receipt.abi (backend receipt (plan c store ifcond) initial) Bool.true)
      else none
    | _,_,_ => none
    else none

/-- Ordinary frontend/compiler/runtime binding. This is not a Lean proof of
the Python or typed-DSL compiler. It maps exact source operands, FMA order and
seeds, FP32 casts, memory effects, and scheduling to the explicit semantics.
It says nothing about real-valued answers. -/
def CompilerBinding (isa : ISA) (c : Shape) (args : Arguments) (receipt : AllocationReceipt c args)
    (scale : BitVec 32) (store ifcond : Bool)
    (backend : AllocationReceipt c args → Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32)) : Prop :=
  ¬Empty c → backend receipt (plan c store ifcond) initial=
    Source.run (wordOps isa) c receipt.abi scale store ifcond initial

theorem success (isa : ISA) (c : Shape) (args : Arguments) (receipt : AllocationReceipt c args)
    (scale : BitVec 32) (store ifcond : Bool)
    (backend : AllocationReceipt c args → Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32))
    (checks : InputChecks c args) (hst : args.store=some store) (hic : args.ifcond=some ifcond)
    (machine : Machine.Domain c receipt.abi)
    (binding : CompilerBinding isa c args receipt scale store ifcond backend initial) :
    run c args (some receipt) backend initial = some
      (result c args receipt.abi (Source.run (wordOps isa) c receipt.abi scale store ifcond initial)
        (decide (¬Empty c))) := by
  classical
  by_cases he : Empty c
  · have hsource : Source.run (wordOps isa) c receipt.abi scale store ifcond initial=initial := by
      unfold Source.run
      exact if_pos he
    simp [run,checks,hst,hic,hsource,he]
  · simp [run,checks,hst,hic,machine,binding he,he]

theorem failed_input (c : Shape) (args : Arguments) (allocation : Option (AllocationReceipt c args))
    (backend : AllocationReceipt c args → Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32)) (bad : ¬InputChecks c args) :
    run c args allocation backend initial=none := by simp [run,bad]

theorem empty_memory (c : Shape) (abi : ABI c) (ops : Ops α) (scale : α)
    (store ifcond : Bool) (initial : Memory α) (he : Empty c) :
    Source.run ops c abi scale store ifcond initial=initial := by
  unfold Source.run
  exact if_pos he

#specNonCircular CompilerBinding avoiding [MathTarget.output]
end Host
#axiomsClean Host.success
#axiomsClean Host.failed_input
#axiomsClean Host.empty_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
