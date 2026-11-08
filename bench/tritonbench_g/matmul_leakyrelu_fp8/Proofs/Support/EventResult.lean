import bench.tritonbench_g.matmul_leakyrelu_fp8.Contract
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked_Real

/-! Reused complete event execution, bound to actual Tensor addresses. This is
an intermediate lemma layer; public source execution must additionally connect
the readable shared typed quotation to this event computation. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
open VeriTile.Triton
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000

def eventCode (r : Request) (c : OutputTensor) (trials : Nat) : Code α Unit :=
  computationCode (arguments r) (outputDescriptor r c) trials

def eventRun (ops : Operations α) (r : Request) (c : OutputTensor) (trials : Nat)
    (read : Nat → α) (scratch : Memory α) : Unit × Memory α :=
  (eventCode r c trials).eval ops (initial r read scratch)

theorem event_complete (ops : Operations α) (r : Request) (c : OutputTensor)
    (h : Contract r c) (trials : Nat) (read : Nat → α) (scratch : Memory α) :
    eventRun ops r c trials read scratch =
      ((),resultMemory ops (arguments r) (outputDescriptor r c) (initial r read scratch)) :=
  eval_computation ops (arguments r) (outputDescriptor r c) (layout r c h) trials (initial r read scratch)

/-- Independent caller-coordinate recurrence. Only actual logical input
addresses and fixed K8 lane primitives occur; there is no final-state access. -/
def logicalAcc (ops : Operations α) (r : Request) (read : Nat → α)
    (i j : Nat) : Nat → α
  | 0 => ops.zero
  | n+1 => block ops
    (fun k => if n*64+k.val < r.A.columns then logical read r.A i (n*64+k.val) else ops.zero)
    (fun k => if n*64+k.val < r.A.columns then logical read r.B (n*64+k.val) j else ops.zero)
    (logicalAcc ops r read i j n)

def logicalReference (ops : Operations α) (r : Request) (read : Nat → α) (i j : Nat) : α :=
  if r.A.columns = 0 then ops.zero else
    ops.castHalf (activate ops (r.activation == "leaky_relu")
      (logicalAcc ops r read i j (arguments r).problem.kTiles))

theorem reference_logical (ops : Operations α) (r : Request) (c : OutputTensor)
    (h : Contract r c) (read : Nat → α) (scratch : Memory α) (i j : Nat)
    (hi : i < r.A.rows) (hj : j < r.B.columns) :
    reference ops (arguments r) (initial r read scratch) i j = logicalReference ops r read i j := by
  have acc (n : Nat) : referenceAcc ops (arguments r) (initial r read scratch) i j n =
      logicalAcc ops r read i j n := by
    induction n with
    | zero => rfl
    | succ n ih =>
      simp only [referenceAcc, logicalAcc, ih]
      congr 1
      · funext k
        change (if n*64+k.val < r.A.columns then _ else _) = _
        split_ifs with hk
        · exact initial_a r h read scratch i _ hi hk
        · rfl
      · funext k
        change (if n*64+k.val < r.A.columns then _ else _) = _
        split_ifs with hk
        · exact initial_b r h read scratch _ j hk hj
        · rfl
  simp only [reference, logicalReference, acc]
  rfl

theorem event_value (ops : Operations α) (r : Request) (c : OutputTensor)
    (h : Contract r c) (trials : Nat) (read : Nat → α) (scratch : Memory α)
    (i j : Nat) (hi : i < r.A.rows) (hj : j < r.B.columns) :
    (eventRun ops r c trials read scratch).2 ((outputDescriptor r c).address i j) =
      logicalReference ops r read i j := by
  rw [event_complete ops r c h]
  simp only [Prod.snd]
  rw [result_reference ops (arguments r) (outputDescriptor r c)
    (layout r c h) (initial r read scratch) i j hi hj]
  exact reference_logical ops r c h read scratch i j hi hj

/-- Ordinary local machine-event simulation composes over the complete code. -/
theorem native_event_complete (abi : LocalABI α σ ops) (r : Request) (c : OutputTensor)
    (h : Contract r c) (trials : Nat) (state : σ) :
    ((eventCode r c trials).native abi state).1 = () ∧
    abi.observe ((eventCode r c trials).native abi state).2 =
      resultMemory ops (arguments r) (outputDescriptor r c) (abi.observe state) := by
  have hh := native_refines abi (eventCode r c trials) state
  rw [eventCode, eval_computation ops (arguments r) (outputDescriptor r c) (layout r c h)] at hh
  constructor
  · simpa only [eventCode] using congrArg (fun z : Unit × Memory α => z.1) hh
  · simpa only [eventCode] using congrArg (fun z : Unit × Memory α => z.2) hh

#specNonCircular logicalReference avoiding [eventCode,eventRun,resultMemory]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
