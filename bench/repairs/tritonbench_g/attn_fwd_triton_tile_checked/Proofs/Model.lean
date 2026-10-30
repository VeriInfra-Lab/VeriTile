import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Graph

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural
open AttnFwdTriton.TypedExecution Graph
noncomputable section

abbrev Ptr := RegionName × Nat
abbrev Matrix (m n : Nat) := TileIndex [m,n] → Expr
abbrev Rows := TileIndex [128] → Expr

structure Fixed (BD : Nat) where
  (context keyStride valueStride startM : Nat)
  rows : TileIndex [128] → Nat
  keys : TileIndex [64] → Nat
  keyBase : TileIndex [BD,64] → Ptr
  scaleBase : Ptr
  valueBase : TileIndex [64,BD] → Ptr
  output : TileIndex [128,BD] → Ptr
  query : Matrix 128 BD
  queryScale : Expr

structure Carry (BD : Nat) where
  max : Rows
  denom : Rows
  accum : Matrix 128 BD

structure Cursor (BD : Nat) where
  key : TileIndex [BD,64] → Ptr
  scale : Ptr
  value : TileIndex [64,BD] → Ptr

/-- Only original input loads, fixed tile inputs and local instruction nodes
occur in this recurrence. It does not call a source/interpreter execution. -/
def keyTile (p : Config) (BD : Nat) (read : Ptr → Expr) (f : Fixed BD)
    (cur : Nat) (ptr : Cursor BD) : Matrix BD 64 := fun i =>
  if decide (f.keys (i.2.1,PUnit.unit) < f.context-cur) && decide (i.1.val<p.headDim)
  then read (ptr.key i) else .zero

def valueTile (p : Config) (BD : Nat) (read : Ptr → Expr) (f : Fixed BD)
    (cur : Nat) (ptr : Cursor BD) : Matrix 64 BD := fun i =>
  if decide (f.keys (i.1,PUnit.unit) < f.context-cur) && decide (i.2.1.val<p.headDim)
  then read (ptr.value i) else .zero

def scores (p : Config) (BD : Nat) (read : Ptr → Expr) (f : Fixed BD)
    (cur : Nat) (ptr : Cursor BD) (causal : Bool) : Matrix 128 64 := fun i =>
  if (if causal then decide (cur+f.keys (i.2.1,PUnit.unit)≤f.rows (i.1,PUnit.unit))
      else Bool.true) && decide (f.keys (i.2.1,PUnit.unit)<f.context-cur)
  then .product (some .fp32) .fp32 Bool.true
    (.product (some .fp32) .fp32 Bool.false
      (.scoreDot (some .fp32) "tf32" [] 128 BD 64 f.query (keyTile p BD read f cur ptr) i)
      f.queryScale) (read ptr.scale)
  else .negInf

def nextMax (score : Matrix 128 64) (old : Carry BD) : Rows := fun i =>
  .binary (some .fp32) .fp32 .max (old.max i)
    (.reduction (some .fp32) Bool.true [128,64] ⟨1,by decide⟩ Bool.false score i)

def probability (score : Matrix 128 64) (m : Rows) : Matrix 128 64 := fun i =>
  .unary (some .fp32) .exp2 (.binary (some .fp32) .fp32 .sub (score i) (m (i.1,PUnit.unit)))

def alpha (old : Carry BD) (m : Rows) : Rows := fun i =>
  .unary (some .fp32) .exp2 (.binary (some .fp32) .fp32 .sub (old.max i) (m i))

def advance (f : Fixed BD) (ptr : Cursor BD) : Cursor BD :=
  ⟨fun i => ((ptr.key i).1,(ptr.key i).2+64*f.keyStride),
   (ptr.scale.1,ptr.scale.2+1),
   fun i => ((ptr.value i).1,(ptr.value i).2+64*f.valueStride)⟩

def update (p : Config) (BD : Nat) (read : Ptr → Expr) (f : Fixed BD)
    (cur : Nat) (ptr : Cursor BD) (causal : Bool) (old : Carry BD) : Carry BD :=
  let score := scores p BD read f cur ptr causal
  let m := nextMax score old
  let prob := probability score m
  let a := alpha old m
  ⟨m,
   fun i => .fma (some .fp32) .fp32 (old.denom i) (a i)
     (.reduction (some .fp32) Bool.false [128,64] ⟨1,by decide⟩ Bool.false prob i),
   fun i => .fma (some .fp32) .fp32 (old.accum i) (a (i.1,PUnit.unit))
     (.cast (some .fp32) .fp16 .fp32
       (.typedDot (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16} [] 128 64 BD
         (fun j => .cast (some .fp32) .fp32 .fp16 (prob j))
         (fun j => .cast (some .fp32) .fp32 .fp16 (valueTile p BD read f cur ptr j)) i))⟩

def initialCarry (BD : Nat) : Carry BD := ⟨fun _ => .negInf,fun _ => .one,fun _ => .zero⟩

def fold (p : Config) (BD : Nat) (read : Ptr → Expr) (f : Fixed BD)
    (stop : Nat) (causal : Bool) (cur : Nat) (ptr : Cursor BD) (old : Carry BD) : Carry BD :=
  if h : cur < stop then
    fold p BD read f stop causal (cur+64) (advance f ptr) (update p BD read f cur ptr causal old)
  else old
termination_by stop-cur

/-- Final source division and explicit BF16 store conversion. -/
def output (c : Carry BD) : Matrix 128 BD := fun i =>
  .cast (some .fp32) .fp32 .bf16
    (.binary (some .fp32) .fp32 .div (c.accum i) (c.denom (i.1,PUnit.unit)))

#specNonCircular fold avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
