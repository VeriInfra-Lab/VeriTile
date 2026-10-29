import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ReductionSchedules

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
noncomputable section

/-- Integer fields of ATen's actual ReduceConfig and the selected output
coordinate. These include all choices affecting operation order. TensorIterator
splitting supplies one such descriptor per 32-bit sub-iterator in source order. -/
structure CudaReduceConfig where
  count : Nat
  blockWidth : Nat
  blockHeight : Nat
  ctas : Nat
  inputX : Nat
  inputY : Nat
  inputCTA : Nat
  step : Nat
  vectorizeInput : Bool
  alignmentShift : Nat
  outputLane : Nat
  outputWarp : Nat
  lastCTA : Nat

def CudaReduceConfig.inputIndex (c : CudaReduceConfig) (x y cta : Nat) : Nat :=
  x*c.inputX+y*c.inputY+cta*c.inputCTA

/-- The index map comes from the selected sub-iterator's offsets. It maps
indices only; it cannot compute or assume any floating reduction result. -/
structure CudaReducePass (width : Nat) where
  config : CudaReduceConfig
  inputIndex : Fin config.count → Fin width

def CudaReducePass.read (pass : CudaReducePass width) (identity : Word) (i : Nat) : ReductionTree width :=
  if h : i<pass.config.count then .input (pass.inputIndex ⟨i,h⟩) else .identity identity

def foldIndices (pass : CudaReducePass width) (identity : Word) (indices : List Nat)
    (initial : ReductionTree width) : ReductionTree width :=
  indices.foldl (fun acc i => .combine acc (pass.read identity i)) initial

/-- CUDA thread_reduce_impl: four separate accumulators, strided unrolled
loads including the tail, followed by (((acc0+acc1)+acc2)+acc3). No reordering
or flattening of these partial sums is performed. -/
def cudaThreadStrided (pass : CudaReducePass width) (identity : Word) (x y cta : Nat) : ReductionTree width :=
  let c := pass.config
  let start := c.inputIndex x y cta
  let accumulator := fun slot =>
    foldIndices pass identity (((List.range c.count).map (fun k => start+(slot+4*k)*c.step)).filter (·<c.count)) (.identity identity)
  [1,2,3].foldl (fun a slot => .combine a (accumulator slot)) (accumulator 0)

/-- CUDA input_vectorized_thread_reduce_impl: alignment head, four vector
accumulators, scalar tail in accumulator zero, then ordered accumulator merge.
The alignment shift is kept even for nonzero sub-iterator storage offsets. -/
def cudaThreadVectorized (pass : CudaReducePass width) (identity : Word) (x y cta : Nat) : ReductionTree width :=
  let c := pass.config
  let start := c.inputIndex x y cta
  let tailOwner := (c.inputY=0 ∨ y=0) ∧ (c.inputCTA=0 ∨ cta=0)
  let shift := if c.alignmentShift=0 then 0 else 4-c.alignmentShift
  let remaining := c.count-shift
  let head := if c.alignmentShift>0 ∧ c.alignmentShift≤x ∧ x<4 ∧ tailOwner
    then ReductionTree.combine (.identity identity) (pass.read identity (x-c.alignmentShift))
    else .identity identity
  let accumulator := fun slot =>
    foldIndices pass identity
      ((((List.range c.count).map (fun k => start+k*c.step)).filter (fun i => i*4+3<remaining)).map
        (fun i => shift+i*4+slot)) (if slot=0 then head else .identity identity)
  let tail := remaining-remaining%4+x
  let first := if tailOwner ∧ tail<remaining then
    ReductionTree.combine (accumulator 0) (pass.read identity (shift+tail)) else accumulator 0
  [1,2,3].foldl (fun a slot => .combine a (accumulator slot)) first

def cudaThread (pass : CudaReducePass width) (identity : Word) (x y cta : Nat) : ReductionTree width :=
  if pass.config.vectorizeInput then cudaThreadVectorized pass identity x y cta
  else cudaThreadStrided pass identity x y cta

def descendingOffsets (dimension : Nat) : List Nat :=
  ((List.range dimension).map (fun power => 2^power)).filter (·<dimension) |>.reverse

/-- Shared-memory halves followed by CUDA's decreasing shfl.down offsets.
The last expression is read at x=0 for a reduced x dimension. -/
def cudaBlockX (dimension epoch y : Nat) (values : Nat → ReductionTree width) : Nat → ReductionTree width :=
  (descendingOffsets dimension).foldl (fun current offset x =>
    if offset≥32 then
      if x<offset then .combine (current x)
        (.transfer (.shared epoch (epoch+1) (4*(x+offset+y*dimension)) (x+offset) x) (current (x+offset)))
      else current x
    else
      let source := if x%32+offset<min dimension 32 then x+offset else x
      .combine (current x) (.transfer (.shuffle source x offset) (current source))) values

def cudaBlockY (dimension blockWidth epoch x : Nat) (values : Nat → ReductionTree width) : Nat → ReductionTree width :=
  (descendingOffsets dimension).foldl (fun current offset y =>
    if y<offset then .combine (current y)
      (.transfer (.shared epoch (epoch+1) (4*(x+(y+offset)*blockWidth)) (y+offset) y) (current (y+offset)))
    else current y) values

def cudaLocalCTA (pass : CudaReducePass width) (identity : Word) (cta : Nat) : ReductionTree width :=
  let c := pass.config
  let xs := fun y => if c.inputX=0 then cudaThread pass identity c.outputLane y cta
    else cudaBlockX c.blockWidth 0 y (fun x => cudaThread pass identity x y cta) 0
  if c.inputY=0 then xs c.outputWarp else cudaBlockY c.blockHeight c.blockWidth 1 c.outputLane xs 0

/-- global_reduce stages CTA partials by CTA index, then the last finisher
reads those slots in strided increasing order. Completion order chooses the
executing CTA, not the floating reduction order. Its y reduction precedes x. -/
def cudaGlobal (pass : CudaReducePass width) (identity : Word) : ReductionTree width :=
  let c := pass.config
  let ctaValue := fun cta => ReductionTree.transfer (.global 2 3 (4*cta) cta c.lastCTA) (cudaLocalCTA pass identity cta)
  let perThread := fun x y =>
    let start := if c.inputX=0 then y else x+y*c.blockWidth
    let stride := if c.inputX=0 then c.blockHeight else c.blockWidth*c.blockHeight
    (((List.range c.ctas).map (fun k => start+k*stride)).filter (·<c.ctas)).foldl
      (fun acc cta => ReductionTree.combine acc (ctaValue cta)) (.identity identity)
  let ys := fun x => cudaBlockY c.blockHeight c.blockWidth 3 x (perThread x) 0
  if c.inputX=0 then ys c.outputLane else cudaBlockX c.blockWidth 4 0 ys 0

def cudaPass (pass : CudaReducePass width) (identity : Word) : ReductionTree width :=
  if pass.config.inputCTA=0 then cudaLocalCTA pass identity 0 else cudaGlobal pass identity

/-- with_32bit_indexing visits sub-iterators in their actual order. Float32
accumulation uses combine(previous_output, current_value) before each store. -/
def cudaPasses (passes : List (CudaReducePass width)) (identity : Word) : ReductionTree width :=
  match passes with
  | [] => .identity identity
  | first::rest => rest.foldl (fun acc pass => .combine acc (cudaPass pass identity)) (cudaPass first identity)

theorem cuda_passes_equiv (h : Obligations n r) (passes : List (CudaReducePass width))
    (identity : Word) (op : ScalarBinary) (xs ys : Fin width → Word) (inputs : ∀ i, xs i=ys i) :
    ((cudaPasses passes identity).graph op).eval n xs=((cudaPasses passes identity).graph op).eval r ys :=
  reduction_equiv h _ op xs ys inputs

#axiomsClean cuda_passes_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
