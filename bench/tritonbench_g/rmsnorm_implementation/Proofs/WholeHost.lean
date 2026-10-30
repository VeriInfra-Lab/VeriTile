import bench.tritonbench_g.rmsnorm_implementation.Proofs.WholeStructure

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section
set_option maxHeartbeats 5000000

/-- Only ordinary torch.empty_like facts; no numerical result or launch result.
The allocation runs after unpacking x.shape and before expanding weight strides. -/
structure Allocated (α : Type) (p : Parameters) (entry : Memory α)
    (preexisting : Nat → Prop) where
  memory : Memory α
  inputPreexisting : preexisting p.input.allocation
  weightPreexisting : preexisting p.weights.allocation
  outputFresh : ¬ preexisting p.output.allocation
  outputShape : p.output.shape = p.input.shape
  outputDtype : p.output.dtype = p.input.dtype
  outputDevice : p.output.device = p.input.device
  outputUnique : OutputUnique p
  frame : ∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset

structure Domain (p : Parameters) where
  weightLength : Nat
  inputShape : p.input.shape = [p.batches,p.rows,p.width]
  weightShape : p.weights.shape = [weightLength]
  weightLongEnough : p.width ≤ weightLength
  inputFP32 : p.input.dtype = .fp32
  weightFP32 : p.weights.dtype = .fp32
  sameDevice : p.weights.device = p.input.device
  epsilonPositive : 0 < p.eps
  capacity : Nat → Nat
  inputBounds : ∀ b m k, b < p.batches → m < p.rows → k < p.width →
    inputOffset p b m k < capacity p.input.allocation
  weightBounds : ∀ k, k < p.width → weightOffset p k < capacity p.weights.allocation
  outputBounds : ∀ b m k, b < p.batches → m < p.rows → k < p.width →
    outputOffset p b m k < capacity p.output.allocation

/-- The exact list of 2D grid coordinates, permuted only to choose scheduling. -/
structure Schedule (p : Parameters) where
  order : List (Nat × Nat)
  complete : order.Perm (programGrid p)

def Allocated.separation (a : Allocated α p entry preexisting) : Separation p where
  input := fun same => a.outputFresh (same ▸ a.inputPreexisting)
  weights := fun same => a.outputFresh (same ▸ a.weightPreexisting)

/-- Source shape unpack, empty_like, stride expansion, grid launch and return.
The allocated object represents the allocator post-state; allocation freshness
and its frame are the only facts imported about the allocator. -/
def entryRun (Q : Numerics α) (red : Reduction) (p : Parameters)
    (a : Allocated α p entry preexisting) (u : Nat → α) (s : Schedule p) : Option (Memory α) :=
  match p.input.shape with
  | [b,m,k] =>
      if p.weights.shape.length = 1 then
        some (executeGrid Q red {p with batches:=b,rows:=m,width:=k} u s.order a.memory)
      else none
  | _ => none

theorem entryRun_eq (Q : Numerics α) (red : Reduction) (p : Parameters)
    (a : Allocated α p entry preexisting) (d : Domain p) (u : Nat → α) (s : Schedule p) :
    entryRun Q red p a u s = some (executeGrid Q red p u s.order a.memory) := by
  simp only [entryRun,d.inputShape,d.weightShape,List.length_cons,List.length_nil,
    Nat.zero_add,if_true]

/-- Both passes and all programs read the original physical input/weight cells;
input-weight aliasing is allowed, and every preexisting allocation is preserved. -/
theorem entry_result (Q : Numerics α) (red : Reduction) (p : Parameters)
    (a : Allocated α p entry preexisting) (d : Domain p) (u : Nat → α) (s : Schedule p) :
    ∃ final, entryRun Q red p a u s = some final ∧
      (∀ b m k, b < p.batches → m < p.rows → k < p.width →
        final p.output.allocation (outputOffset p b m k) =
          logicalValue Q red p.width p.eps
            (fun j => entry p.input.allocation (inputOffset p b m j))
            (fun j => entry p.weights.allocation (weightOffset p j)) k) ∧
      (∀ allocation offset, Outside p allocation offset →
        final allocation offset = a.memory allocation offset) ∧
      (∀ allocation offset, preexisting allocation → final allocation offset = entry allocation offset) := by
  obtain ⟨values,frame⟩ := grid_result Q red p a.separation a.outputUnique u s.order s.complete a.memory
  refine ⟨_,entryRun_eq Q red p a d u s,?_,frame,?_⟩
  · intro b m k hb hm hk
    rw [values b m k hb hm hk]
    congr 2
    · funext j
      exact a.frame _ _ a.inputPreexisting
    · funext j
      exact a.frame _ _ a.weightPreexisting
  · intro allocation offset existed
    rw [frame, a.frame allocation offset existed]
    intro b m k hb hm hk same
    exact a.outputFresh (same.1 ▸ existed)

/-- Literal host order, including failure timing. Invalid x rank fails before
allocation; invalid weight rank fails during stride expansion after allocation. -/
def hostOperations : List String :=
  ["batch,M,K=x.shape", "out=torch.empty_like(x)",
   "rmsnorm_triton[(batch,M)](x,rms_weights,out,*x.stride(),*rms_weights.stride(),*out.stride(),N_SIZE=K,eps=eps,BLOCK_N_SIZE=4096,num_warps=16)",
   "return out"]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
