import bench.repairs.tritonbench_g.triton_argmax.Proofs.HostExecution

/-! Request-specific metadata binds the exact branch, dtype argument, block
policy and returned object. Generic allocation layouts alone are insufficient
for an API theorem. These records contain frontend/tensor metadata only. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.RequestBinding
open Source Memory Host HostExecution

structure BlockPolicy where
  sqrtCeil : Nat → Nat
  nextPowerOfTwo : Nat → Nat
  sqrtPositive : ∀ n,0 < n → 0 < sqrtCeil n
  nextPositive : ∀ n,0 < n → 0 < nextPowerOfTwo n
  nextCovers : ∀ n,n ≤ nextPowerOfTwo n

def block (policy : BlockPolicy) (size : Nat) : Nat := policy.nextPowerOfTwo (policy.sqrtCeil size)
def blocks (policy : BlockPolicy) (size : Nat) : Nat := (size+block policy size-1)/block policy size
def midBlock (policy : BlockPolicy) (size : Nat) : Nat := policy.nextPowerOfTwo (blocks policy size)

def fp32Format : MidFormat := ⟨"torch.float32",4,by omega⟩
def requestedFormat (r : Request) : Option MidFormat :=
  match r.dtype with
  | .omitted => some fp32Format
  | .supported d => some d
  | .unsupported _ => none

structure GlobalMetadata {M : Type} {d : MidFormat} (mem : Primitives M) (before : M) (r : Request)
    (policy : BlockPolicy) (setup : GlobalSetup mem before r.input d) : Prop where
  branch : r.dim=none
  dtypeResolved : requestedFormat r=some d
  nonempty : 0 < r.input.numel
  inputDtype : r.input.dtype="torch.float32"
  blockValue : setup.c.block=block policy r.input.numel
  blockCount : setup.c.blocks=blocks policy r.input.numel
  midCount : setup.c.midBlock=midBlock policy r.input.numel
  returnedShape : setup.output.returned.shape=outputShape r.input.shape none r.keepdim
  outputDevice : setup.output.returned.storage.owner=r.input.storage.owner
  midValueDevice : setup.p.midValue.owner=r.input.storage.owner
  midIndexDevice : setup.p.midIndex.owner=r.input.storage.owner
  firstComplete : ∀ pid,pid < setup.c.blocks → (setup.c.firstTree pid).Complete
  secondComplete : setup.c.secondTree.Complete

theorem global_shape_product (shape : List Nat) (keep : Bool) :
    (outputShape shape none keep).prod=1 := by
  cases keep <;> simp [outputShape]

theorem GlobalMetadata.outputSize {M : Type} {d : MidFormat} {mem : Primitives M} {before : M}
    {r : Request} {policy : BlockPolicy} {setup : GlobalSetup mem before r.input d}
    (metadata : GlobalMetadata mem before r policy setup) : setup.output.returned.numel=1 := by
  unfold Tensor.numel
  rw [metadata.returnedShape]
  exact global_shape_product _ _

theorem global_coverage_from_request {M : Type} {d : MidFormat} (mem : Primitives M) (before : M) (r : Request)
    (policy : BlockPolicy) (setup : GlobalSetup mem before r.input d)
    (metadata : GlobalMetadata mem before r policy setup) : Math.GlobalCoverage setup.c := by
  apply Host.global_coverage setup.c
  · rw [setup.size]
    exact metadata.nonempty
  · rw [metadata.blockValue]
    exact policy.nextPositive _ (policy.sqrtPositive _ metadata.nonempty)
  · rw [metadata.blockValue,metadata.blockCount,setup.size]
    rfl
  · rw [metadata.blockCount,metadata.midCount]
    exact policy.nextCovers _
  · exact metadata.firstComplete
  · exact metadata.secondComplete

structure DimMetadata {M : Type} (mem : Primitives M) (before : M) (r : Request)
    (setup : DimSetup mem before r.input) where
  requestedDim : Int
  ax : Nat
  branch : r.dim=some requestedDim
  normalized : normalizeDim r.input.shape.length requestedDim=some ax
  valid : ax < r.input.shape.length
  inputDtype : r.input.dtype="torch.float32"
  outer : setup.c.outer=(r.input.shape.take ax).prod
  extent : setup.c.extent=(r.input.shape[ax]?.getD 0)
  inner : setup.c.inner=(r.input.shape.drop (ax+1)).prod
  nonempty : 0 < setup.c.extent
  returnedShape : setup.output.returned.shape=outputShape r.input.shape (some ax) r.keepdim
  outputDevice : setup.output.returned.storage.owner=r.input.storage.owner
  treeComplete : ∀ row inner tile,(setup.c.tree row inner tile).Complete

theorem dimension_shape_product (shape : List Nat) (ax : Nat) (valid : ax < shape.length) (keep : Bool) :
    (outputShape shape (some ax) keep).prod=(shape.take ax).prod*(shape.drop (ax+1)).prod := by
  induction shape generalizing ax with
  | nil => simp at valid
  | cons head tail ih =>
    cases ax with
    | zero => cases keep <;> simp [outputShape]
    | succ ax =>
      have valid' : ax < tail.length := by simpa using valid
      have rest := ih ax valid'
      cases keep <;>
        simpa [outputShape,List.take_succ_cons,List.drop_succ_cons,Nat.mul_assoc] using congrArg (head*·) rest

theorem DimMetadata.outputSize {M : Type} {mem : Primitives M} {before : M}
    {r : Request} {setup : DimSetup mem before r.input}
    (metadata : DimMetadata mem before r setup) : setup.output.returned.numel=setup.c.outer*setup.c.inner := by
  unfold Tensor.numel
  rw [metadata.returnedShape,dimension_shape_product _ _ metadata.valid,metadata.outer,metadata.inner]

/-- Independent of dtype: the original dimension branch never reads its dtype
argument. The bound metadata determines the actual M/N/K values. -/
theorem shape_factorization (shape : List Nat) (ax : Nat) (valid : ax < shape.length) :
    (shape.take ax).prod*(shape[ax]?.getD 0)*(shape.drop (ax+1)).prod=shape.prod := by
  have split := List.take_append_drop ax shape
  have products := congrArg (fun xs : List Nat => xs.prod) split
  dsimp only at products
  rw [List.prod_append,List.drop_eq_getElem_cons valid,List.prod_cons] at products
  simpa [List.getElem?_eq_getElem valid,Nat.mul_assoc] using products

theorem global_logical_output {M : Type} {d : MidFormat} (mem : Primitives M) (before : M) (r : Request)
    (policy : BlockPolicy) (setup : GlobalSetup mem before r.input d)
    (metadata : GlobalMetadata mem before r policy setup) (ops : Scalars d) :
    outputWord mem (globalRun mem ops before r.input setup) setup.output 0=
      BitVec.ofNat 64 (Source.globalIndex ops setup.c (logicalWord mem before r.input)) := by
  have mapped := setup.output.linearCoordinates 0 (by rw [metadata.outputSize]; omega)
  unfold outputWord
  rw [mapped]
  have observed := global_returned mem ops before r.input setup
  simpa only [setup.output.sameOffset,setup.output.zeroOffset] using observed

theorem dimension_logical_output {M : Type} {d : MidFormat} (mem : Primitives M)
    (ops : Scalars d) (before : M) (r : Request) (setup : DimSetup mem before r.input)
    (metadata : DimMetadata mem before r setup) (row inner : Nat)
    (hr : row < setup.c.outer) (hi : inner < setup.c.inner) :
    outputWord mem (dimensionRun mem ops before r.input setup) setup.output (dimOutputOffset setup.c row inner)=
      BitVec.ofNat 64 (dimIndex ops setup.c (logicalWord mem before r.input) row inner) := by
  have bound : dimOutputOffset setup.c row inner < setup.output.returned.numel := by
    rw [metadata.outputSize]
    exact Native.dim_output_bound _ _ _ _ hr hi
  unfold outputWord
  rw [setup.output.linearCoordinates _ bound]
  have observed := dimension_returned mem ops before r.input setup row inner hr hi
  simpa only [setup.output.sameOffset,setup.output.zeroOffset,Nat.zero_add] using observed

#axiomsClean global_coverage_from_request
#axiomsClean shape_factorization
#axiomsClean global_logical_output
#axiomsClean dimension_logical_output
end VeriTile.Bench.Repairs.TritonArgmax.RequestBinding
