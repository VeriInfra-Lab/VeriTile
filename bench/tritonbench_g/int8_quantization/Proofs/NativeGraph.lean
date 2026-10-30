import bench.tritonbench_g.int8_quantization.Proofs.FPHost
import VeriTile.Triton.Float.TermModel

/-! Scalar FP32 instruction semantics and the compiler-selected ordered maximum
reduction. No primitive takes an output tensor or implements a whole block.
The tree records operand order and padding; its validity checks every lane.
IEEE rounding, FTZ, comparisons, literal conversion and FP-to-int conversion
are the instruction model boundary, not unproved kernel transformations. -/
namespace VeriTile.Bench.Repairs.Int8Quantization.NativeGraph
open VeriTile.Triton FP.Structural Revision002 Words Direct
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
set_option maxHeartbeats 1000000

/-- The selected target's scalar instructions. `div32` retains the production
division lowering and FTZ policy; `int8RTZ` includes the final signed-byte cast.
The load/store ABI already represents all four FP32 bytes without conversion. -/
structure Instructions where
  literal32 : ℝ → Word
  abs32 : Word → Word
  mul32 : Word → Word → Word
  div32 : Word → Word → Word
  add32 : Word → Word → Word
  max32 : Word → Word → Word
  min32 : Word → Word → Word
  less32 : Word → Word → Bool
  le32 : Word → Word → Bool
  int8RTZ : Word → Int
  maxIdentity : Word

/-- These are structural compiler data, never arbitrary whole-vector numeric
callbacks. `padding` and the ordered tree preserve inserted identity operands.
`ReductionPlan.valid` requires every source lane exactly once. -/
abbrev Plans := (n : Nat) → ReductionPlan n

def maximum (op : Instructions) (xs : Fin n → Word) : ReductionTree n → Word
  | .input j => xs j
  | .zero => op.maxIdentity
  | .add left right => op.max32 (maximum op xs left) (maximum op xs right)

def reduce (op : Instructions) (plans : Plans) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (xs : FP.Structural.Values Word .real shape) :
    FP.Structural.Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => maximum op
      (fun j => xs (TileShape.insertAxisIndex shape ax out j))
      (plans (TileShape.axisDim shape ax)).tree
  · exact fun out => maximum op
      (fun j => xs (TileShape.replaceAxisIndex shape ax out j))
      (plans (TileShape.axisDim shape ax)).tree

/-- Only reached operations are replaced. Source floating context `none` and
`some fp32` both resolve to the selected FP32 instructions. -/
def algebra (rest : Algebra Word) (op : Instructions) (plans : Plans) : Algebra Word :=
  { rest with
    literal := fun _ _ => op.literal32
    unary := fun _ tag x => match tag with
      | .abs _ => op.abs32 x
      | _ => rest.unary (some .fp32) tag x
    binary := fun _ _ tag x y => match tag with
      | .mul => op.mul32 x y
      | .div => op.div32 x y
      | .add => op.add32 x y
      | .max => op.max32 x y
      | _ => rest.binary (some .fp32) .real tag x y
    fp32Load := id
    reduceMax := fun _ => reduce op plans }

def comparisons (op : Instructions) : Model.Primitives Word :=
  ⟨op.less32,op.le32,op.int8RTZ,op.min32⟩

/-- Independently stated numerical graph; the maximum expands only binary
scalar calls, with the compiler's explicit operand order. -/
def numerics (op : Instructions) (plans : Plans) : Floating.Numerics Word where
  abs := op.abs32
  mul := op.mul32
  div := op.div32
  add := op.add32
  max := op.max32
  min := op.min32
  le := op.le32
  lt := op.less32
  maximum := fun {n} xs => maximum op xs (plans n).tree
  int8 := op.int8RTZ

def constants (op : Instructions) (factor : ℝ) : Floating.Constants Word :=
  ⟨op.literal32 0.0,op.literal32 1.0,op.literal32 (-1.0),op.literal32 0.5,
    op.literal32 127.0,op.literal32 (-127.0),
    op.literal32 340282346638528859811704183484516925440.0,op.literal32 factor⟩

def block (op : Instructions) (plans : Plans) (query : Bool) (factor : ℝ)
    (input : Fin n → Word) : (Fin n → Int) × Word :=
  Floating.run (constants op factor) (numerics op plans) query input

theorem context (rest : Algebra Word) (op : Instructions) (plans : Plans)
    (p : Parameters) (entry : State Word) :
    FPTrace.Context (algebra rest op plans) p entry := ⟨rfl,fun _ => rfl⟩

theorem numerics_eq (rest : Algebra Word) (op : Instructions) (plans : Plans) :
    FPTrace.numerics (algebra rest op plans) (comparisons op)=numerics op plans := by
  rfl

theorem local_agreement (rest : Algebra Word) (op : Instructions) (plans : Plans)
    (p : Parameters) (query : Bool) (input : Fin n → Word) :
    Floating.Local (FPTrace.constants (algebra rest op plans) p)
      (FPTrace.numerics (algebra rest op plans) (comparisons op))
      (numerics op plans) query input := by
  rw [numerics_eq]
  constructor <;> intros <;> rfl

/-- The complete typed source terminates successfully and both actual stores
read back the independent ordered scalar graph. No arithmetic agreements,
finite-input restrictions or output-equality premises are assumed. -/
theorem source_success_graph (rest : Algebra Word) (op : Instructions) (plans : Plans)
    (p : Parameters) (positive : 0 < p.width) (entry : State Word)
    (i : Lane p) (active : Model.active p entry i) :
    ∃ final, FP.Structural.exec (Model.backend (algebra rest op plans) (comparisons op))
        (source p) entry=some final ∧
      final.mem outRegion (Model.offset p entry i)=
        .mk .int ((block op plans p.query p.factor
          (FPTrace.inputs (algebra rest op plans) p entry)).1 i.1) ∧
      final.mem scaleRegion (Model.group p entry)=
        .mk .real ((block op plans p.query p.factor
          (FPTrace.inputs (algebra rest op plans) p entry)).2) := by
  refine ⟨finalState (algebra rest op plans) (comparisons op) p positive entry,?_,?_⟩
  · exact source_run _ _ _ _ _
  · exact FPTrace.source_fp _ _ p positive entry (context rest op plans p entry)
      (numerics op plans) (local_agreement rest op plans p p.query _) i active

/-- Original caller storage, masked exactly as the source. Loads preserve raw
FP32 words; padding is the actual literal zero word. -/
def inputs (c : Request) (op : Instructions) (original : Memory)
    (w : Schedule.Program) : Fin (p c).width → Word :=
  fun j => if Grid.active (p c) w (j,PUnit.unit) then
    inputValue c original (Grid.offset (p c) w (j,PUnit.unit)) else op.literal32 0.0

def target (c : Request) (op : Instructions) (plans : Plans)
    (original : Memory) (w : Schedule.Program) : (Fin (p c).width → Int) × Word :=
  if channels c.x.metadata=0 then (fun _ => 0,op.literal32 0.0) else
    block op plans c.query c.factor (inputs c op original w)

structure Values (c : Request) (op : Instructions) (plans : Plans)
    (original : Memory) (output scales : Tensor) (result : Memory) : Prop where
  output : ∀ w, w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c), Grid.active (p c) w i →
    result (output.base+Grid.offset (p c) w i)=BitVec.ofInt 8 ((target c op plans original w).1 i.1)
  scales : ∀ w, w∈Schedule.programs 0 (count c) →
    Bytes.read32 result (scales.base+4*w.index)=(target c op plans original w).2

theorem target_eq (c : Request) (rest : Algebra Word) (op : Instructions) (plans : Plans)
    (original : Memory) (w : Schedule.Program) :
    reference c (algebra rest op plans) (numerics op plans) original w=
      target c op plans original w := by
  unfold reference target
  split_ifs
  · norm_num [algebra]
  · rfl

theorem values (c : Request) (rest : Algebra Word) (op : Instructions) (plans : Plans)
    (original before : Heap) (out : Allocation c before) (output scales : Tensor) (result : Memory)
    (observed : Direct.Values c (algebra rest op plans) (comparisons op)
      before.memory (layout c out) output scales result)
    (ext : Extension original before) (live : InputLive c original) :
    Values c op plans original.memory output scales result := by
  have observed := numerical_values c (algebra rest op plans) (comparisons op)
    original before out output scales result observed ext live
    (fun w _ _ => context rest op plans _ _) (numerics op plans)
    (fun w _ _ => local_agreement rest op plans (p c) c.query _)
  constructor
  · simpa only [target_eq] using observed.output
  · simpa only [target_eq] using observed.scales

#specNonCircular target avoiding [source,FP.Structural.exec,algebra,FPTrace.numerics,
  finalState,quantized,savedScale,typedRun,after,pair]
#axiomsClean values

#specNonCircular block avoiding [source,FP.Structural.exec,algebra,FPTrace.numerics,
  finalState,quantized,savedScale]
#axiomsClean source_success_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.NativeGraph
