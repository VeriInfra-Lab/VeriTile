import bench.tritonbench_g.batched_vecmat_mult.Proofs.ValueModel

/-! Natural-integer logical host metadata. Allocation remains an explicit
primitive boundary: all three original fp32/CUDA allocation requests execute
before the K/M/N guards, and each can fail. No native RNG distribution, allocator,
freshness or Python arbitrary-type semantics is claimed by this module. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.Host
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Arguments where
  (M N K BM BN BK : Nat)
  numWarps : Nat := 4
  numStages : Nat := 1

inductive Init where
  | randn | zeros
  deriving DecidableEq, Repr

/-- Each request is specifically for CUDA float32 storage, as hard-coded by
the original wrapper. The callback interprets shape, initialization and failure. -/
structure AllocationRequest where
  init : Init
  shape : List Nat
  deriving DecidableEq, Repr

structure AllocationResult where
  region : RegionName
  memory : Memory

abbrev Allocator := AllocationRequest → Memory → Except String AllocationResult

inductive Error where
  | allocationA (message : String) | allocationB (message : String) | allocationOutput (message : String)
  | zeroBlockK | innerDivisibility | zeroBlockM | rowDivisibility | zeroBlockN | colDivisibility
  deriving DecidableEq, Repr

structure Launch where
  parameters : Parameters
  memory : Memory
  (numWarps numStages : Nat)

def allocateAs (allocate : Allocator) (request : AllocationRequest) (mem : Memory)
    (error : String → Error) : Except Error AllocationResult :=
  match allocate request mem with
  | .error message => .error (error message)
  | .ok result => .ok result

/-- Call order is deliberately allocation A, allocation B, zero-allocation
output, K remainder/guard, M remainder/guard, N remainder/guard, launch. -/
def prepare (allocate : Allocator) (a : Arguments) (mem : Memory) : Except Error Launch := do
  let va ← allocateAs allocate ⟨.randn,[a.M,a.K]⟩ mem .allocationA
  let vb ← allocateAs allocate ⟨.randn,[a.M,a.N,a.K]⟩ va.memory .allocationB
  let vo ← allocateAs allocate ⟨.zeros,[a.M,a.N]⟩ vb.memory .allocationOutput
  if a.BK = 0 then .error .zeroBlockK else do
    if a.K % a.BK ≠ 0 then .error .innerDivisibility else do
      if a.BM = 0 then .error .zeroBlockM else do
        if a.M % a.BM ≠ 0 then .error .rowDivisibility else do
          if a.BN = 0 then .error .zeroBlockN else do
            if a.N % a.BN ≠ 0 then .error .colDivisibility else
              pure { parameters := ⟨va.region,vb.region,vo.region,a.M,a.N,a.K,a.BM,a.BN,a.BK⟩
                     memory := vo.memory, numWarps := a.numWarps, numStages := a.numStages }

structure Prepared (allocate : Allocator) (a : Arguments) (mem : Memory) (l : Launch) : Prop where
  allocations : ∃ va vb vo,
    allocate ⟨.randn,[a.M,a.K]⟩ mem = .ok va ∧
    allocate ⟨.randn,[a.M,a.N,a.K]⟩ va.memory = .ok vb ∧
    allocate ⟨.zeros,[a.M,a.N]⟩ vb.memory = .ok vo ∧
    l.parameters.A = va.region ∧ l.parameters.B = vb.region ∧
    l.parameters.output = vo.region ∧ l.memory = vo.memory
  dimensions : l.parameters.M = a.M ∧ l.parameters.N = a.N ∧ l.parameters.K = a.K
  blocks : l.parameters.BM = a.BM ∧ l.parameters.BN = a.BN ∧ l.parameters.BK = a.BK
  positive : 0 < a.BM ∧ 0 < a.BN ∧ 0 < a.BK
  divisible : a.K % a.BK = 0 ∧ a.M % a.BM = 0 ∧ a.N % a.BN = 0
  tuning : l.numWarps = a.numWarps ∧ l.numStages = a.numStages

theorem prepare_fields (allocate : Allocator) (a : Arguments) (mem : Memory) (l : Launch)
    (h : prepare allocate a mem = .ok l) : Prepared allocate a mem l := by
  unfold prepare allocateAs at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' (first | split at h | simp at h)
  all_goals cases h
  all_goals constructor <;> simp_all
  all_goals first | omega | aesop

theorem allocation_failure_precedes_guards (allocate : Allocator) (a : Arguments) (mem : Memory)
    (message : String) (h : allocate ⟨.randn,[a.M,a.K]⟩ mem = .error message) :
    prepare allocate a mem = .error (.allocationA message) := by
  simp [prepare, allocateAs, h, bind, Except.bind]

theorem zero_inner_block_after_allocations (allocate : Allocator) (a : Arguments) (mem : Memory)
    (va vb vo : AllocationResult)
    (ha : allocate ⟨.randn,[a.M,a.K]⟩ mem = .ok va)
    (hb : allocate ⟨.randn,[a.M,a.N,a.K]⟩ va.memory = .ok vb)
    (ho : allocate ⟨.zeros,[a.M,a.N]⟩ vb.memory = .ok vo)
    (hk : a.BK = 0) : prepare allocate a mem = .error .zeroBlockK := by
  simp [prepare, allocateAs, ha, hb, ho, hk, bind, Except.bind]

#axiomsClean prepare_fields
#axiomsClean allocation_failure_precedes_guards
#axiomsClean zero_inner_block_after_allocations
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs.Host
