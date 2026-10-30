import bench.tritonbench_g.triton_softmax.Proofs.FP
import bench.tritonbench_g.softmax_triton1.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.TritonSoftmax.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000

open VeriTile.Bench.Repairs.SoftmaxTriton1
abbrev Word := SoftmaxTriton1.NativeGraph.Word
abbrev Node := SoftmaxTriton1.NativeGraph.Node
abbrev atom := SoftmaxTriton1.NativeGraph.atom
abbrev negInf := SoftmaxTriton1.NativeGraph.negInf
abbrev log2e := SoftmaxTriton1.NativeGraph.log2e
abbrev Instructions := SoftmaxTriton1.NativeGraph.Instructions
abbrev butterfly := SoftmaxTriton1.NativeGraph.butterfly

/-- V is the actual sizePerThread from the compiled blocked layout. The
original wrapper's fixed four-warps configuration is retained for every B. -/
structure Profile where
  vector : Nat
  allowed : vector=1 ∨ vector=2 ∨ vector=4

def threads (B : Nat) := 32*TritonSoftmax.numWarps B

def registerIndex (V B thread slot : Nat) : Nat :=
  (slot/V)*(threads B*V)+thread*V+slot%V

def owner (V B i : Nat) : Nat := (i/V)%threads B

def warpDistances (V B : Nat) : List Nat :=
  [16,8,4,2,1].filter (fun d => d < min (B/V) 32)

def groupDistances (V B : Nat) : List Nat :=
  [8,4,2,1].filter (fun d => d < min (B/(32*V)) (TritonSoftmax.numWarps B))

/-- One register left-fold, seeded by slot zero without an extra neutral
operation. Slots are in compiled blocked-layout register order. -/
def registerFold (op : Word → Word → Word) (V B : Nat)
    (leaf : Nat → Nat → Word) (thread : Nat) : Word :=
  (List.range (max 1 (B/threads B)-1)).foldl
    (fun accum slot => op accum (leaf (registerIndex V B thread (slot+1)) thread))
    (leaf (registerIndex V B thread 0) thread)

/-- Every shuffle stage reads its predecessor. Multi-warp reductions store
lane-zero warp results to shared memory, reduce them in the first warp, and
broadcast its lane-zero result. No floating associativity is used. -/
def reduceWords (op : Word → Word → Word) (V B : Nat)
    (leaf : Nat → Nat → Word) (thread : Nat) : Word :=
  let warp := butterfly op (warpDistances V B) (registerFold op V B leaf)
  if B/V≤32 then warp (thread%max (B/V) 1)
  else butterfly op (groupDistances V B) (fun w => warp (w*32)) 0

def reduceNodes (op : Word → Word → Word) (V B : Nat) (values : Nat → Node) : Node :=
  reduceWords op V B values

def reduceValues (op : Word → Word → Word) (V : Nat) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => reduceNodes op V (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨j,h⟩) else atom 0)
  · exact fun out => reduceNodes op V (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩) else atom 0)

def algebra (p : Instructions) (V : Nat) : Algebra Node where
  literal := fun _ _ _ => atom 0
  negInf := atom negInf
  binary := fun _ _ op a b => match op with
    | .sub => fun t => p.subRN (a t) (b t)
    | .div => fun t => p.divFull (a t) (b t)
    | _ => atom 0
  unary := fun _ op a => match op with
    | .exp => fun t => p.exp2Approx (p.mulRN (a t) log2e)
    | _ => atom 0
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom 0
  fromInt := fun _ _ => atom 0
  fp32Bits := fun w => atom w.bits
  fp32Load := id
  reduceMax := fun _ => reduceValues p.maxNum V
  reduceSum := fun _ => reduceValues p.addRN V

def masked (n : Nat) (xs : Nat → Word) (j : Nat) : Word :=
  if j<n then xs j else negInf

def maximum (p : Instructions) (V n B : Nat) (xs : Nat → Word) : Node :=
  reduceWords p.maxNum V B (fun j _ => if j<B then masked n xs j else 0)

def numerator (p : Instructions) (V n B : Nat) (xs : Nat → Word) (j t : Nat) : Word :=
  p.exp2Approx (p.mulRN (p.subRN (masked n xs j) (maximum p V n B xs t)) log2e)

def denominator (p : Instructions) (V n B : Nat) (xs : Nat → Word) : Node :=
  reduceWords p.addRN V B (fun j t => if j<B then numerator p V n B xs j t else 0)

/-- Independent native register/shuffle/shared-memory graph at caller words.
Padding is the exact FP32 -infinity word. Exp is log2(e) multiply then ex2;
division remains div.full and no reciprocal or FMA rewrite is used. -/
def row (p : Instructions) (V n B : Nat) (xs : Nat → Word) (i : Nat) : Word :=
  p.divFull (numerator p V n B xs i (owner V B i))
    (denominator p V n B xs (owner V B i))

theorem source_graph_lowering (p : Instructions) (V n B : Nat) (xs : Nat → Word)
    (i : Nat) (active : i<B) :
    (Floating.formula (algebra p V) n B (fun j => atom (xs j)) i) (owner V B i)=
      row p V n B xs i := by
  have maxeq :
      (algebra p V).reduceMax (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false
        (fun j => if j.1.val<n then atom (xs j.1.val) else (algebra p V).negInf) PUnit.unit=
      maximum p V n B xs := by
    change reduceNodes p.maxNum V B _=maximum p V n B xs
    unfold maximum reduceNodes
    congr 1
    funext j t
    by_cases hj : j<B <;> by_cases hn : j<n <;> simp [hj,hn,TileShape.axisDim,
      TileShape.insertAxisIndex,algebra,atom,SoftmaxTriton1.NativeGraph.atom,masked]
  unfold Floating.formula SoftmaxTriton1.Floating.formula
  rw [dif_pos active]
  unfold SoftmaxTriton1.Floating.rowExpression
  rw [maxeq]
  change p.divFull _ _=row p V n B xs i
  unfold row denominator numerator
  congr 1
  · by_cases hi : i<n <;> simp [algebra,masked,hi,atom,SoftmaxTriton1.NativeGraph.atom]
  · change reduceWords p.addRN V B _ _=reduceWords p.addRN V B _ _
    congr 1
    funext j t
    by_cases hj : j<B
    · by_cases hn : j<n <;>
        simp [reduceValues,TileShape.axisDim,TileShape.insertAxisIndex,hj,hn,
          algebra,atom,SoftmaxTriton1.NativeGraph.atom,masked,numerator]
    · simp [reduceValues,TileShape.axisDim,hj,atom,SoftmaxTriton1.NativeGraph.atom]

abbrev InstructionAgreement := SoftmaxTriton1.NativeGraph.InstructionAgreement
abbrev instructions_equal := SoftmaxTriton1.NativeGraph.instructions_equal

theorem row_agreement (a b : Instructions) (h : InstructionAgreement a b)
    (V n B : Nat) (xs : Nat → Word) (i : Nat) :
    row a V n B xs i=row b V n B xs i := by rw [instructions_equal a b h]

#specNonCircular row avoiding [Floating.formula,Floating.rowExpression,algebra]
#axiomsClean source_graph_lowering
#axiomsClean row_agreement
end
end VeriTile.Bench.Repairs.TritonSoftmax.NativeGraph
