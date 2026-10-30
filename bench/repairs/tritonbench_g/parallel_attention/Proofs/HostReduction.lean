import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostPlans

namespace VeriTile.Bench.Repairs.ParallelAttention.TorchHost
noncomputable section

/-- Ordinary selected ReduceConfig metadata. These fields describe source
routing, never arithmetic results. Each launch reduces one interval belonging
to the current output cell; TensorIterator splitting can select several launches. -/
structure Launch where
  first : Nat
  inputs : Nat
  width : Nat
  height : Nat
  reduceX : Bool
  reduceY : Bool
  ctas : Nat
  vectorInput : Bool
  alignmentShift : Nat
  accumulate : Bool
  deriving Repr,DecidableEq

def Launch.xStride (d : Launch) : Nat := if d.reduceX then 1 else 0
def Launch.yStride (d : Launch) : Nat := if d.reduceY then (if d.reduceX then d.width else 1) else 0
def Launch.ctaStride (d : Launch) : Nat := (if d.reduceX then d.width else 1)*(if d.reduceY then d.height else 1)
def Launch.step (d : Launch) : Nat := d.ctaStride*d.ctas
def Launch.start (d : Launch) (x y cta : Nat) : Nat := x*d.xStride+y*d.yStride+cta*d.ctaStride

/-- Increasing loop indices, with no hidden reduction callback. -/
def indices (first step stop : Nat) : List Nat :=
  (List.range ((stop-first+step-1)/step)).map (fun i => first+i*step)

def chain (initial : SumPlan) (terms : List SumPlan) : SumPlan := terms.foldl SumPlan.add initial

def inputChain (initial : SumPlan) (offset first step stop : Nat) : SumPlan :=
  chain initial ((indices first step stop).map (fun i => .input (offset+i)))

/-- Four independent scalar accumulators, then the exact 0+1,+2,+3 combine
order of Reduce.cuh thread_reduce_impl. -/
def scalarThread (d : Launch) (x y cta : Nat) : SumPlan :=
  let accum := fun i => inputChain .zero d.first (d.start x y cta+i*d.step) (4*d.step) d.inputs
  chain (accum 0) [accum 1,accum 2,accum 3]

/-- Vector-input header, four lane accumulators and the scalar tail retain
row-pointer alignment. Only the designated y/CTA owner reads header/tail. -/
def vectorThread (d : Launch) (x y cta : Nat) : SumPlan :=
  let ownsTail := (!d.reduceY || decide (y=0)) && decide (cta=0)
  let head := if d.alignmentShift=0 then 0 else 4-d.alignmentShift
  let header := if d.alignmentShift>0 && decide (d.alignmentShift ≤ x ∧ x<4) && ownsTail
    then SumPlan.add .zero (.input (d.first+x-d.alignmentShift)) else .zero
  let remaining := d.inputs-head
  let accum := fun i => chain (if i=0 then header else .zero)
    ((indices (d.start x y cta) d.step (remaining/4)).map
      (fun j => .input (d.first+head+j*4+i)))
  let tail := remaining-remaining%4+x
  let firstAccum := if ownsTail && decide (tail<remaining)
    then SumPlan.add (accum 0) (.input (d.first+head+tail)) else accum 0
  chain firstAccum [accum 1,accum 2,accum 3]

def threadPlan (d : Launch) (x y cta : Nat) : SumPlan :=
  if d.vectorInput then vectorThread d x y cta else scalarThread d x y cta

/-- Source shared/shuffle reduction at offsets width/2,...,1. The active
output lane is zero, so only valid owner dependencies are evaluated. -/
def halve : Nat → (Nat → SumPlan) → Nat → SumPlan
  | 0,values,index => values index
  | n+1,values,index =>
      halve n (fun i => .add (values i) (values (i+2^n))) index

def blockPlan (d : Launch) (cta : Nat) : SumPlan :=
  let rows := fun y => if d.reduceX then halve d.width.log2 (fun x => threadPlan d x y cta) 0
    else threadPlan d 0 y cta
  if d.reduceY then halve d.height.log2 rows 0 else rows 0

/-- The final CTA first combines its assigned staging slots in increasing
order, then reduces Y and X in the opposite order to the first stage, exactly
as global_reduce specifies. Completion order never selects arithmetic order. -/
def globalPlan (d : Launch) : SumPlan :=
  let staged := fun x y =>
    let start := if d.reduceX then x+y*d.width else y
    let step := if d.reduceX then d.width*d.height else d.height
    chain .zero ((indices start step d.ctas).map (blockPlan d))
  let columns := fun x => halve d.height.log2 (fun y => staged x y) 0
  if d.reduceX then halve d.width.log2 columns 0 else columns 0

def launchPlan (d : Launch) : SumPlan := if 1<d.ctas then globalPlan d else blockPlan d 0

/-- With at most four scratch shards and more than one output, the source
selector uses this scalar route. Output vectorization/block dimensions do not
change its expression, including singleton axes removed by coalescing. -/
def scalarOnly (first inputs : Nat) : SumPlan :=
  let accum := fun i => inputChain .zero first i 4 inputs
  chain (accum 0) [accum 1,accum 2,accum 3]

theorem scalar_route (d : Launch) (hx : d.reduceX=false) (hy : d.reduceY=false)
    (hv : d.vectorInput=false) (hc : d.ctas=1) : launchPlan d=scalarOnly d.first d.inputs := by
  simp [launchPlan,blockPlan,threadPlan,scalarThread,scalarOnly,
    Launch.start,Launch.step,Launch.ctaStride,Launch.xStride,Launch.yStride,hx,hy,hv,hc]

/-- TensorIterator's ordered split launches preserve the actual output
accumulation flag. FP32 accumulation is old-output + next-launch. -/
def selectPlan (launches : List Launch) : SumPlan :=
  launches.foldl (fun previous d => if d.accumulate then .add previous (launchPlan d) else launchPlan d) .zero

/-- Source launch dimensions are positive powers of two, selected by
set_block_dimension. Global reduction always follows a Y reduction. The
step bound records that every participating reduction owner enters run's
thread_reduce branch. These are integer routing conditions only. -/
def Launch.sane (d : Launch) : Bool :=
  decide (0<d.inputs ∧ d.width=2^d.width.log2 ∧ d.height=2^d.height.log2 ∧
    d.width*d.height≤512 ∧ 0<d.ctas ∧ d.step≤d.inputs ∧ d.alignmentShift<4) &&
  (d.ctas==1 || d.reduceY) &&
  (!d.vectorInput || (d.reduceX && decide (128≤d.inputs)))

/-- A source-specific finite structural checker. It checks input indices and
multiplicity, not values, numerical results, execution success or final memory.
The external compiler contract binds the actual ReduceConfig/routing to these
metadata and each combine to one scalar add. -/
def admissible (n : Nat) (launches : List Launch) : Bool :=
  launches.all Launch.sane && decide ((selectPlan launches).leaves.Perm (List.range n))

theorem admissible_complete (n : Nat) (launches : List Launch)
    (checked : admissible n launches=true) : (selectPlan launches).Covers n := by
  change (selectPlan launches).leaves.Perm (List.range n)
  simp only [admissible,Bool.and_eq_true] at checked
  have coverage : decide ((selectPlan launches).leaves.Perm (List.range n)) = true := checked.2
  exact of_decide_eq_true coverage

theorem selected_real (n : Nat) (launches : List Launch)
    (checked : admissible n launches=true) (x : Nat → ℝ) :
    (selectPlan launches).eval realOps x=∑ i∈Finset.range n,x i :=
  SumPlan.real_sum _ n (admissible_complete n launches checked) x

/-- A producer supplies actual ordered TensorIterator sublaunch metadata,
and the finite checker derives coverage. No arbitrary sum tree is accepted. -/
structure Selection (n : Nat) where
  launches : List Launch
  checked : admissible n launches=true

def Selection.tree {n : Nat} (d : Selection n) : SumPlan := selectPlan d.launches

theorem Selection.complete {n : Nat} (d : Selection n) : d.tree.Covers n :=
  admissible_complete n d.launches d.checked

/-- Only the source-level Real bridge uses this total plan family. Native
normalizers use NativePlans.normTree and public host sums use Selection.tree. -/
def referencePlans : Plans where
  dotTree := fun _ => DotPlan.linear
  dotComplete := fun _ => DotPlan.linear_complete
  sumTree := fun _ => serialSum
  sumComplete := fun _ => serialSum_complete

#axiomsClean admissible_complete
#axiomsClean selected_real
#axiomsClean Selection.complete
#axiomsClean scalar_route
end
end VeriTile.Bench.Repairs.ParallelAttention.TorchHost
