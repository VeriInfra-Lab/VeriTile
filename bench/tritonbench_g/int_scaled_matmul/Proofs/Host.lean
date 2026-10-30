import bench.tritonbench_g.int_scaled_matmul.Proofs.ViewSafety
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Host
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
abbrev Word := ScaledWord.Word
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

inductive DType where
  | int8 | int32 | fp32
  deriving DecidableEq
/-- A rank-two tensor view; capacity is in elements from its data pointer.
Region identities must respect physical storage aliasing. -/
structure Tensor where
  (objectId device : Nat)
  (region : RegionName)
  (rows columns rowStride columnStride capacity : Nat)
  (dtype : DType)
structure Config where
  (blockRows blockColumns blockInner groupRows : Nat)
structure Arguments where
  (a b c : Tensor)
  (config : Config)

def shapeCompatible (x : Arguments) : Prop :=
  x.a.columns=x.b.rows ∧ x.c.rows=x.a.rows ∧ x.c.columns=x.b.columns
instance (x : Arguments) : Decidable (shapeCompatible x) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))
def emptyOutput (x : Arguments) : Prop := x.a.rows=0 ∨ x.b.columns=0
instance (x : Arguments) : Decidable (emptyOutput x) := inferInstanceAs (Decidable (_ ∨ _))
def tensorBounds (t : Tensor) := ViewBounds t.rows t.columns t.rowStride t.columnStride t.capacity

def blockParameters (x : Arguments) : BlockWord.Parameters :=
  ⟨x.a.region,x.b.region,x.c.region,x.a.rows,x.b.columns,x.a.columns,
    x.a.rowStride,x.a.columnStride,x.b.rowStride,x.b.columnStride,x.c.rowStride,x.c.columnStride,
    x.config.blockRows,x.config.blockColumns,x.config.blockInner,x.config.groupRows⟩
def scaledParameters (x : Arguments) (scale : Tensor) : ScaledWord.Parameters :=
  ⟨x.a.region,x.b.region,x.c.region,scale.region,x.a.rows,x.b.columns,x.a.columns,
    x.a.rowStride,x.a.columnStride,x.b.rowStride,x.b.columnStride,x.c.rowStride,x.c.columnStride,
    scale.rowStride,scale.columnStride,
    x.config.blockRows,x.config.blockColumns,x.config.blockInner,x.config.groupRows,
    (x.a.columns+x.config.blockInner-1)/x.config.blockInner,
    decide (x.a.columns%x.config.blockInner=0)⟩
abbrev Index (x : Arguments) := Fin x.a.rows × Fin x.b.columns
def address (x : Arguments) (i : Index x) := i.1.val*x.c.rowStride+i.2.val*x.c.columnStride

/-- Selected source precision and ordinary tensor/layout hypotheses. This
contains no expected result or kernel-correctness assumption. -/
structure Valid (x : Arguments) : Prop where
  shape : shapeCompatible x
  aDtype : x.a.dtype=.int8
  bDtype : x.b.dtype=.int8
  cDtype : x.c.dtype=.int32
  sameDevice : x.a.device=x.b.device ∧ x.a.device=x.c.device
  aBounds : tensorBounds x.a
  bBounds : tensorBounds x.b
  cBounds : tensorBounds x.c
  writable : Function.Injective (address x)
  aSeparate : x.a.region≠x.c.region
  bSeparate : x.b.region≠x.c.region
  blockRows : 0<x.config.blockRows
  blockColumns : 0<x.config.blockColumns
  blockInner : 0<x.config.blockInner
  groupRows : 0<x.config.groupRows
structure ScaleValid (x : Arguments) (scale : Tensor) : Prop where
  shape : scale.rows=x.a.rows ∧ scale.columns=1
  dtype : scale.dtype=.fp32
  sameDevice : scale.device=x.a.device
  bounds : tensorBounds scale
  separate : scale.region≠x.c.region

def Int8Storage (t : Tensor) (s : State Word) : Prop :=
  ∀ r c, r<t.rows → c<t.columns → ∃ z : Int,
    s.mem t.region (r*t.rowStride+c*t.columnStride)=Cell.mk .int z ∧ BlockWord.wrap 8 z=z
def FP32Storage (t : Tensor) (s : State Word) : Prop :=
  ∀ r c, r<t.rows → c<t.columns → ∃ z : Word,
    s.mem t.region (r*t.rowStride+c*t.columnStride)=Cell.mk .real z

/-- Source host control: shape assertion, empty-output early return, launch,
and return of exactly the supplied C object. -/
def int_matmul_kernel (rest : Algebra Word) (x : Arguments) (s : State Word) : Option (Tensor × State Word) :=
  if shapeCompatible x then
    if emptyOutput x then some (x.c,s)
    else (BlockWord.grid rest (blockParameters x) s).map (fun final => (x.c,final))
  else none

def int_scaled_matmul_kernel (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (s : State Word) : Option (Tensor × State Word) :=
  if shapeCompatible x then
    if emptyOutput x then some (x.c,s)
    else (ScaledWord.grid rest cfg u (scaledParameters x scale) s).map (fun final => (x.c,final))
  else none

theorem geometry_valid (x : Arguments) (h : Valid x) : Geometry.Valid (BlockWord.geometry (blockParameters x)) :=
  ⟨h.blockRows,h.blockColumns,h.groupRows,h.writable⟩
theorem scaled_geometry_valid (x : Arguments) (scale : Tensor) (h : Valid x) :
    Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)) :=
  ⟨h.blockRows,h.blockColumns,h.groupRows,h.writable⟩

theorem launch_cover (x : Arguments) (scale : Tensor) (h : Valid x) :
    (scaledParameters x scale).inner≤(scaledParameters x scale).steps*(scaledParameters x scale).blockInner :=
  LogicalSum.ceil_cover _ _ h.blockInner

theorem launch_even (x : Arguments) (scale : Tensor) (h : Valid x) :
    (scaledParameters x scale).even=Bool.true →
    (scaledParameters x scale).inner=(scaledParameters x scale).steps*(scaledParameters x scale).blockInner := by
  intro he
  change decide (x.a.columns%x.config.blockInner=0)=Bool.true at he
  exact LogicalSum.ceil_exact _ _ h.blockInner (of_decide_eq_true he)

theorem shape_rejected (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (s : State Word) (h : ¬shapeCompatible x) :
    int_matmul_kernel rest x s=none ∧ int_scaled_matmul_kernel rest cfg u x scale s=none := by
  simp [int_matmul_kernel,int_scaled_matmul_kernel,h]

theorem empty_return (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (s : State Word) (hs : shapeCompatible x) (he : emptyOutput x) :
    int_matmul_kernel rest x s=some (x.c,s) ∧
    int_scaled_matmul_kernel rest cfg u x scale s=some (x.c,s) := by
  simp [int_matmul_kernel,int_scaled_matmul_kernel,hs,he]

#axiomsClean geometry_valid
#axiomsClean scaled_geometry_valid
#axiomsClean launch_cover
#axiomsClean launch_even
#axiomsClean shape_rejected
#axiomsClean empty_return
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Host
