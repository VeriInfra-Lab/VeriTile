import bench.tritonbench_g.triton_attention.Proofs.NativeScalar
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Target
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 4000000

abbrev Tensor := Nat → Nat → Nat → Graph
abbrev Rows := Nat → Nat → Graph

def scaleWord (scale : ℝ) : Graph :=
  if scale=0 then .word 0 else if scale=1 then .word 0x3f800000 else .literal scale

def scaleMul (one : Bool) (scale : ℝ) (x : Graph) : Graph :=
  if scale=0 then .mul x (.word 0) else if scale=1 then
    if one then x else .mul (.word 0x3f800000) x
  else .mul (.literal scale) x

def scaleSubtract (one : Bool) (scale : ℝ) (x m : Graph) : Graph :=
  if one && decide (scale=1) then .sub x m else .fma (scaleWord scale) x (.neg m)

def exp (x : Graph) : Graph := .exp2 (.mul x (.word 0x3fb8aa3b))
def half (x : Graph) : Graph := .widen (.narrow x)
def pad (size : Nat) (f : Nat → Graph) (i : Nat) := if i<size then f i else .word 0

structure Inputs where
  q : Tensor
  k : Tensor
  v : Tensor
  grad : Tensor
structure Acc where
  maximum : Fin 4 → Graph
  denominator : Fin 4 → Graph
  output : Fin 4 → Graph

def initial : Acc := ⟨fun _=>.word 0xff800000,fun _=>.word 0,fun _=>.word 0⟩

def score (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row b j : Nat) : Graph :=
  if b*128+j≤row then scaleMul one scale
    (dot .n128 c.dim (pad c.dim (x.q h row)) (pad c.dim (x.k h (b*128+j))) Bool.false (.word 0))
  else .word 0xff800000

def maximum (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row b : Nat) (a : Acc) (o : Fin 4) : Graph :=
  .max ((Reduction.plan o).eval Graph.max (pad 128 (score c one scale x h row b))) (a.maximum o)
def probability (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row b : Nat) (a : Acc) (j : Nat) : Graph :=
  exp (.sub (score c one scale x h row b j) (maximum c one scale x h row b a (Reduction.owner j)))
def denominator (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row b : Nat) (a : Acc) (o : Fin 4) : Graph :=
  .fma (a.denominator o) (exp (.sub (a.maximum o) (maximum c one scale x h row b a o)))
    ((Reduction.plan o).eval Graph.add (pad 128 (probability c one scale x h row b a)))
def normalized (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row b : Nat) (a : Acc) (j : Nat) : Graph :=
  half (.mul (probability c one scale x h row b a j)
    (.div (.word 0x3f800000) (denominator c one scale x h row b a (Reduction.owner j))))

def advance (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row col b : Nat) (a : Acc) : Acc :=
  ⟨maximum c one scale x h row b a,denominator c one scale x h row b a,
   fun o=>dot (TritonAttention.Whole001.Native.width c.dim) 128
    (pad 128 (normalized c one scale x h row b a)) (pad 128 (fun j=>x.v h (b*128+j) col)) Bool.true
    (.mul (a.output o) (.mul
      (.mul (a.denominator o) (exp (.sub (a.maximum o) (maximum c one scale x h row b a o))))
      (.div (.word 0x3f800000) (denominator c one scale x h row b a o))))⟩

def rounds (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (h row col : Nat) : Nat → Acc
  | 0=>initial
  | n+1=>advance c one scale x h row col n (rounds c one scale x h row col n)
structure Saved where
  output : Tensor
  denominator : Rows
  maximum : Rows

def forward (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (owners : TritonAttention.Whole001.Native.Owners) : Saved :=
  ⟨fun h i d=>half ((rounds c one scale x h i d (i/128+1)).output (Reduction.owner d)),
   fun h i=>(rounds c one scale x h i 0 (i/128+1)).denominator (owners.l h i),
   fun h i=>(rounds c one scale x h i 0 (i/128+1)).maximum (owners.m h i)⟩

structure Prepared where
  gradient : Tensor
  delta : Rows

def preprocess (c : Shape) (owners : TritonAttention.Whole001.Native.Owners) (x : Inputs) (s : Saved) : Prepared :=
  ⟨fun h i d=>.div (x.grad h i d) (s.denominator h i),
   fun h i=>TritonAttention.Whole001.Native.shuffleAdd ops (TritonAttention.Whole001.Native.preMasks c.dim)
    (TritonAttention.Whole001.Native.preLocal ops (s.output h i) (x.grad h i) (s.denominator h i))
    (owners.delta h i%(c.dim/4))⟩

structure BackwardInputs where
  q : Tensor
  k : Tensor
  v : Tensor
  maximum : Rows
  gradient : Tensor
  delta : Rows

def backwardProbability (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h i j : Nat) : Graph :=
  let qk:=if j ≤ i then dot .n128 c.dim (pad c.dim (x.q h i)) (pad c.dim (x.k h j)) Bool.false (.word 0) else .word 0xff800000
  exp (scaleSubtract one scale qk (x.maximum h i))
def backwardDS (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h i j : Nat) : Graph :=
  scaleMul one scale (.mul (backwardProbability c one scale x h i j)
    (dot .n128 c.dim (pad c.dim (x.gradient h i)) (pad c.dim (x.v h j)) Bool.true (.neg (x.delta h i))))
def queryUpdate (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h i d b : Nat) (seed : Graph) : Graph :=
  if b ≤ i/128 then dot (TritonAttention.Whole001.Native.width c.dim) 128
    (pad 128 (fun j=>half (backwardDS c one scale x h i (b*128+j))))
    (pad 128 (fun j=>x.k h (b*128+j) d)) Bool.true seed else seed

def query (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h i d : Nat) : Nat → Graph
  | 0=>.word 0
  | n+1=>queryUpdate c one scale x h i d n (query c one scale x h i d n)
def key (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h j d : Nat) : Graph :=
  half ((List.range (c.blocks-j/128)).foldl (fun seed t=>
    dot (TritonAttention.Whole001.Native.width c.dim) 128
      (pad 128 (fun i=>half (backwardDS c one scale x h ((j/128+t)*128+i) j)))
      (pad 128 (fun i=>x.q h ((j/128+t)*128+i) d)) Bool.true seed) (.word 0))
def value (c : Shape) (one : Bool) (scale : ℝ) (x : BackwardInputs) (h j d : Nat) : Graph :=
  half ((List.range (c.blocks-j/128)).foldl (fun seed t=>
    dot (TritonAttention.Whole001.Native.width c.dim) 128
      (pad 128 (fun i=>half (backwardProbability c one scale x h ((j/128+t)*128+i) j)))
      (pad 128 (fun i=>x.gradient h ((j/128+t)*128+i) d)) Bool.true seed) (.word 0))

def prepared (c : Shape) (one : Bool) (scale : ℝ) (x : Inputs) (owners : TritonAttention.Whole001.Native.Owners) : BackwardInputs :=
  let s:=forward c one scale x owners
  let p:=preprocess c owners x s
  ⟨x.q,x.k,x.v,s.maximum,p.gradient,p.delta⟩

#specNonCircular forward avoiding [Source.forwardRounds,Grid.forwardStep,FP.Structural.exec]
#specNonCircular query avoiding [Source.backwardRounds,Grid.backwardStep,FP.Structural.exec]
#specNonCircular prepared avoiding [Source.forwardRounds,Source.preWrite,Grid.forwardStep,Grid.preprocessStep,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Target
