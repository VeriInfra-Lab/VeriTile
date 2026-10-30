import bench.repairs.tritonbench_g.fused_recurrent_retention.APISource
import bench.repairs.tritonbench_g.fused_recurrent_retention.AddressCheck
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission

instance (x : TensorObject) (current : Nat) : Decidable (x.RawFP32 current) := by
  unfold TensorObject.RawFP32
  infer_instance
instance (s : Shape) : Decidable s.Admitted := by
  unfold Shape.Admitted
  infer_instance

/-- The observed storage base is kept separate from tensor identity and offset. -/
abbrev Bases := Nat → Nat

def lastOffset (x : TensorObject) : Nat :=
  x.view.offset+(x.dimensions 0-1)*x.view.strides 0+
    (x.dimensions 1-1)*x.view.strides 1+(x.dimensions 2-1)*x.view.strides 2+
    (x.dimensions 3-1)*x.view.strides 3

def tensorCheck (current : Nat) (dims : Fin 4 → Nat) (dense : Bool) (x : TensorObject) : Bool :=
  decide (x.RawFP32 current) && decide (x.dimensions=dims) &&
  (if dense then decide (x.view.strides=contiguous4 (dims 0) (dims 1) (dims 2) (dims 3)) else Bool.true) &&
  decide (lastOffset x<x.view.capacity)

def shapeCheck (s : Shape) : Bool :=
  decide (0<s.B ∧ 0<s.H ∧ 0<s.T ∧ s.Admitted)

def pointerRadius (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers) : Nat :=
  ((formed phase seed extra s b).map (SExpr.radius (limits s))).foldr max 0

def windowCheck (base radius capacity : Nat) : Bool :=
  decide (base%4=0 ∧ 4*radius≤base ∧ base+4*radius+4≤2^64 ∧ base+4*capacity≤2^64)

def objectWindow (bases : Bases) (radius : Nat) (x : TensorObject) : Bool :=
  windowCheck (bases x.view.storage) radius x.view.capacity

def forwardObjects (args : ForwardArguments) : List TensorObject :=
  [args.q,args.k,args.v]++args.seed.toList

def forwardCheck (s : Shape) (args : ForwardArguments) (current : Nat) (bases : Bases) : Bool :=
  shapeCheck s &&
  tensorCheck current ![s.B,s.H,s.T,s.DK] Bool.true args.q &&
  tensorCheck current ![s.B,s.H,s.T,s.DK] Bool.true args.k &&
  tensorCheck current ![s.B,s.H,s.T,s.DV] Bool.true args.v &&
  (args.seed.toList.all (tensorCheck current ![s.B,s.H,s.DK,s.DV] Bool.true)) &&
  addressCheck .forward args.seed.isSome args.outputFinal s args.buffers &&
  (forwardObjects args).all (objectWindow bases (pointerRadius .forward args.seed.isSome args.outputFinal s args.buffers))

def savedArguments (ctx : SavedContext) : ForwardArguments :=
  ⟨ctx.q,ctx.k,ctx.v,ctx.initial,0,ctx.outputFinal⟩

def backwardObjects (ctx : SavedContext) (do_ : TensorObject) (final : Option TensorObject) : List TensorObject :=
  forwardObjects (savedArguments ctx)++[do_]++final.toList

def backwardCheck (s : Shape) (ctx : SavedContext) (do_ : TensorObject) (final : Option TensorObject)
    (current : Nat) (bases : Bases) (gradEnabled : Bool) : Bool :=
  forwardCheck s (savedArguments ctx) current bases && !gradEnabled &&
  tensorCheck current ![s.B,s.H,s.T,s.DV] Bool.false do_ &&
  (final.toList.all fun f=>ctx.outputFinal && tensorCheck current ![s.B,s.H,s.DK,s.DV] Bool.false f) &&
  addressCheck .backward ctx.initial.isSome final.isSome s (ctx.buffers do_.view (final.map TensorObject.view)) &&
  (backwardObjects ctx do_ final).all (objectWindow bases
    (pointerRadius .backward ctx.initial.isSome final.isSome s (ctx.buffers do_.view (final.map TensorObject.view))))

/-- Fresh allocation identities are observed from the actual allocator result;
this check runs before submission, including scratch arrays. -/
def allocationCheck (bases : Bases) (radius : Nat) (objects : List (Nat × Nat)) : Bool :=
  objects.all fun x=>windowCheck (bases x.1) radius x.2
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission
