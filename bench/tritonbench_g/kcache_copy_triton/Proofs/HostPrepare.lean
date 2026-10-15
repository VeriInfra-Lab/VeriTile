import bench.tritonbench_g.kcache_copy_triton.Proofs.HostGrid

/-! Logical metadata execution of the original Python wrapper. The 4D
reshape uses the contiguous-chunk stride inference algorithm, retaining view
aliasing when possible and materializing a row-major copy otherwise. Native
Torch allocation/reshape refinement, arbitrary Python argument types, negative
n, native metadata widths, device execution and omitted dtypes remain outside
this model. They are not numerical assumptions. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host
open VeriTile.Triton FP.Structural HostGrid
open SignedCopy (Parameters)
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

structure Tensor where
  region : RegionName
  shape : List Nat
  strides : List Nat
  deriving Repr

structure Inputs where
  key : Tensor
  cache : Tensor
  lengths : Tensor
  tables : Tensor
  keyProfile : Profile
  cacheProfile : Profile
  n : Nat := 1
  newLayout : Bool := Bool.false
  reshapeRegion : RegionName

inductive Error where
  | dtypeMismatch | reshapeAmbiguous | keyRank | tokenBatch
  | missingBatchDimension | incompatibleBatch | missingCacheDimension
  | missingCacheStride | incompatibleNewCache | zeroWidth | missingKeyStride
  | missingTableStride
  deriving DecidableEq, Repr

/-- Standard contiguous strides, including empty dimensions. -/
def contiguousStrides (shape : List Nat) : List Nat :=
  (shape.foldr (fun n acc => (max n 1 * acc.1, acc.1 :: acc.2)) (1,[])).2

/-- Consume one contiguous storage chunk, starting at the last new dimension.
The tuple records remaining dimensions, their consumed product, and the
assigned strides in reverse dimension order. -/
def consume (newRev : List Nat) (target base product : Nat) : List Nat × Nat × List Nat :=
  match newRev with
  | [] => ([],product,[])
  | n :: ns =>
      if product < target ∨ n = 1 then
        let next := consume ns target base (product * n)
        (next.1,next.2.1,(product * base) :: next.2.2)
      else (newRev,product,[])

def inferChunks : List (Nat × Nat) → List Nat → Nat → Nat → Option (List Nat)
  | [], newRev, _, _ => if newRev = [] then some [] else none
  | (n,_) :: oldRev, newRev, base, chunk =>
      let chunk := chunk * n
      let boundary := match oldRev with
        | [] => Bool.true
        | (previousSize,previousStride) :: _ =>
            previousSize != 1 && previousStride != chunk * base
      if boundary then
        let used := consume newRev chunk base 1
        if used.2.1 = chunk then
          let nextBase := match oldRev with | [] => 0 | (_,s) :: _ => s
          (inferChunks oldRev used.1 nextBase 1).map (used.2.2 ++ ·)
        else none
      else inferChunks oldRev newRev base chunk

/-- The source of this finite algorithm is Torch's computeStride_impl, also
exposed in torch._subclasses.fake_impls._compute_stride. This is an explicit
model of that host primitive, not an axiom that a supplied stride is valid. -/
def inferViewStrides (oldShape oldStrides newShape : List Nat) : Option (List Nat) :=
  if oldShape = [] then some (List.replicate newShape.length 1)
  else if oldShape.prod = 0 then
    if oldShape = newShape then some oldStrides else some (contiguousStrides newShape)
  else ((inferChunks (oldShape.zip oldStrides).reverse newShape.reverse
    (oldStrides.getLast?.getD 0) 1).map List.reverse)

structure Reshape where
  tensor : Tensor
  copied : Bool
  deriving Repr

def reshape (key : Tensor) (fresh : RegionName) : Except Error Reshape :=
  match key.shape with
  | [a,b,h,d] =>
      if h * d = 0 then .error .reshapeAmbiguous
      else
        let shape := [a*b,h,d]
        match inferViewStrides key.shape key.strides shape with
        | some strides => .ok ⟨⟨key.region,shape,strides⟩,Bool.false⟩
        | none => .ok ⟨⟨fresh,shape,contiguousStrides shape⟩,Bool.true⟩
  | _ => .ok ⟨key,Bool.false⟩

def warps (headDim : Nat) : Nat := if headDim > 128 then 8 else 4
def batches (tokens n : Nat) : Nat := if n > 1 then tokens / n else tokens

structure Launch where
  reshaped : Reshape
  parameters : Parameters
  grid : HostGrid.Grid
  numWarps : Nat
  suppliedLastStride : Nat
  headDim : Nat

private def get (items : List Nat) (i : Nat) (failure : Error) : Except Error Nat :=
  match items[i]? with | some n => .ok n | none => .error failure

/-- All original guards are evaluated in source order. In particular the
legacy cache branch has no shape/head-dimension check, both initial cache
stride reads occur before the new-layout assertion, and n=0 reaches grid 0.
The last cache stride is passed but deliberately not used by the kernel. -/
def prepare (h : Inputs) : Except Error Launch := do
  if h.keyProfile ≠ h.cacheProfile then throw .dtypeMismatch
  let k ← reshape h.key h.reshapeRegion
  let [tokens,heads,headDim] := k.tensor.shape | throw .keyRank
  if h.n > 1 ∧ tokens % h.n ≠ 0 then throw .tokenBatch
  let bsz := batches tokens h.n
  let lengthBatch ← get h.lengths.shape 0 .missingBatchDimension
  let tableBatch ← get h.tables.shape 0 .missingBatchDimension
  if lengthBatch ≠ tableBatch ∨ tableBatch ≠ bsz then throw .incompatibleBatch
  let blockSize ← get h.cache.shape.reverse 1 .missingCacheDimension
  let oldCs ← get h.cache.strides 2 .missingCacheStride
  let oldCd ← get h.cache.strides 3 .missingCacheStride
  let (x,splitStride,cs,cd) ←
    if h.newLayout then do
      let [_,cacheHeads,splitCount,_,cacheX] := h.cache.shape | throw .incompatibleNewCache
      if cacheHeads ≠ heads ∨ splitCount * cacheX ≠ headDim then throw .incompatibleNewCache
      let newCs ← get h.cache.strides 3 .missingCacheStride
      let newCd ← get h.cache.strides 4 .missingCacheStride
      pure (cacheX,oldCs,newCs,newCd)
    else pure (headDim,0,oldCs,oldCd)
  -- Python forms head_dim // x while constructing grid before evaluating the
  -- positional kernel arguments below.
  if x = 0 then throw .zeroWidth
  let kt ← get k.tensor.strides 0 .missingKeyStride
  let kh ← get k.tensor.strides 1 .missingKeyStride
  let kd ← get k.tensor.strides 2 .missingKeyStride
  let cb ← get h.cache.strides 0 .missingCacheStride
  let ch ← get h.cache.strides 1 .missingCacheStride
  let bts ← get h.tables.strides 0 .missingTableStride
  let btb ← get h.tables.strides 1 .missingTableStride
  pure {
    reshaped := k
    parameters := {
      K := k.tensor.region, KCache := h.cache.region
      blockTables := h.tables.region.cast, lengths := h.lengths.region.cast
      nTokens := h.n, strideKt := kt, strideKh := kh, strideKd := kd
      strideKcb := cb, strideKch := ch, strideKcsplit := splitStride
      strideKcs := cs, strideBts := bts, strideBtb := btb
      blockSize := blockSize, width := x }
    grid := ⟨bsz * h.n,heads,headDim / x⟩
    numWarps := warps headDim
    suppliedLastStride := cd
    headDim := headDim }

/-- Exact first grid dimension for both declared n branches, without fixing
n to one. n=0 retains the actual zero grid behavior. -/
theorem batch_grid (tokens n : Nat) (same : n > 1 → tokens % n = 0) :
    batches tokens n * n = if n = 0 then 0 else tokens := by
  by_cases hn : n > 1
  · have hmod := same hn
    have h := Nat.mod_add_div tokens n
    simp only [hmod, Nat.zero_add] at h
    simpa [batches, hn, show n ≠ 0 by omega, Nat.mul_comm] using h
  · have : n = 0 ∨ n = 1 := by omega
    rcases this with rfl | rfl <;> simp [batches]

theorem reshape_three (region : RegionName) (t h d st sh sd : Nat) (fresh : RegionName) :
    reshape ⟨region,[t,h,d],[st,sh,sd]⟩ fresh =
      .ok ⟨⟨region,[t,h,d],[st,sh,sd]⟩,Bool.false⟩ := rfl

theorem reshape_four_shape (key : Tensor) (fresh : RegionName)
    (a b h d : Nat) (hs : key.shape = [a,b,h,d]) (nonemptyTail : h*d ≠ 0) :
    ∃ result, reshape key fresh = .ok result ∧ result.tensor.shape = [a*b,h,d] := by
  simp only [reshape, hs, nonemptyTail, ↓reduceIte]
  cases inferViewStrides [a,b,h,d] key.strides [a*b,h,d] with
  | none => exact ⟨_,rfl,rfl⟩
  | some strides => exact ⟨_,rfl,rfl⟩

theorem warps_le128 (headDim : Nat) (h : headDim ≤ 128) : warps headDim = 4 := by
  simp [warps, show ¬headDim > 128 by omega]
theorem warps_gt128 (headDim : Nat) (h : headDim > 128) : warps headDim = 8 := by
  simp [warps,h]

#axiomsClean batch_grid
#axiomsClean reshape_three
#axiomsClean reshape_four_shape
#axiomsClean warps_le128
#axiomsClean warps_gt128
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host
