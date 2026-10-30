import bench.repairs.tritonbench_g.softmax_triton3.Proofs.FP
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeGraph
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section
open Classical
set_option maxHeartbeats 4000000
abbrev Word := BitVec 32
abbrev Node := SoftmaxTriton1.NativeGraph.Node
abbrev Instructions := SoftmaxTriton1.NativeGraph.Instructions
abbrev Profile := SoftmaxTriton1.NativeGraph.Profile
abbrev atom := SoftmaxTriton1.NativeGraph.atom
abbrev negInf := SoftmaxTriton1.NativeGraph.negInf
abbrev log2e := SoftmaxTriton1.NativeGraph.log2e
abbrev owner := SoftmaxTriton1.NativeGraph.owner
abbrev reduceWords := SoftmaxTriton1.NativeGraph.reduceWords
abbrev reduceValues := SoftmaxTriton1.NativeGraph.reduceValues
abbrev maximum := SoftmaxTriton1.NativeGraph.maximum
abbrev masked := SoftmaxTriton1.NativeGraph.masked

def algebra (p : Instructions) (V : Nat) : Algebra Node :=
  { SoftmaxTriton1.NativeGraph.algebra p V with
    binary := fun _ _ op a b => match op with
      | .sub => fun t => p.subRN (a t) (b t)
      | .add => fun t => p.addRN (a t) (b t)
      | .div => fun t => p.divFull (a t) (b t)
      | _ => atom 0 }

/-- The mask addition remains after subtraction, exactly as the source and
native instructions. No exp-splitting, max-shift move or reassociation is used. -/
def numerator (p : Instructions) (V n B : Nat) (xs : Nat → Word)
    (ms : Option (Nat → Word)) (j t : Nat) : Word :=
  let delta := p.subRN (masked n xs j) (maximum p V n B xs t)
  let shifted := match ms with
    | none => delta
    | some ms => p.addRN delta (if j<n then ms j else 0)
  p.exp2Approx (p.mulRN shifted log2e)

def denominator (p : Instructions) (V n B : Nat) (xs : Nat → Word)
    (ms : Option (Nat → Word)) : Node :=
  reduceWords p.addRN V B (fun j t => if j<B then numerator p V n B xs ms j t else 0)

/-- Independent raw-word native graph. registerFold/butterfly/shared-memory
exchange are imported unchanged from reviewed softmax1 revision002 helpers. -/
def row (p : Instructions) (V : Nat) : Formula Word := fun n B xs ms i =>
  p.divFull (numerator p V n B xs ms i (owner V B i))
    (denominator p V n B xs ms (owner V B i))

theorem source_graph_lowering (p : Instructions) (V n B : Nat) (xs : Nat → Word)
    (ms : Option (Nat → Word)) (i : Nat) (active : i<B) :
    (Floating.formula (algebra p V) n B (fun j => atom (xs j))
      (ms.map (fun f j => atom (f j))) i) (owner V B i)=row p V n B xs ms i := by
  have maxeq : (algebra p V).reduceMax (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false
      (Floating.inputRow (algebra p V) n B (fun j => atom (xs j.val))) PUnit.unit=
      maximum p V n B xs := by
    change SoftmaxTriton1.NativeGraph.reduceNodes p.maxNum V B _=maximum p V n B xs
    unfold maximum SoftmaxTriton1.NativeGraph.maximum SoftmaxTriton1.NativeGraph.reduceNodes
    congr 1
    funext j t
    by_cases hj : j<B <;> by_cases hn : j<n <;>
      simp [hj,hn,Floating.inputRow,TileShape.axisDim,TileShape.insertAxisIndex,algebra,
        SoftmaxTriton1.NativeGraph.algebra,atom,SoftmaxTriton1.NativeGraph.atom,
        masked,SoftmaxTriton1.NativeGraph.masked]
  unfold Floating.formula
  rw [dif_pos active]
  unfold Floating.expression
  rw [maxeq]
  cases ms with
  | none =>
    change p.divFull _ _=row p V n B xs none i
    unfold row denominator numerator
    congr 1
    · by_cases hi : i<n <;> simp [algebra,SoftmaxTriton1.NativeGraph.algebra,Floating.inputRow,
        masked,SoftmaxTriton1.NativeGraph.masked,hi,atom,SoftmaxTriton1.NativeGraph.atom]
    · change reduceWords p.addRN V B _ _=reduceWords p.addRN V B _ _
      congr 1
      funext j t
      by_cases hj : j<B <;> by_cases hn : j<n <;>
        simp [SoftmaxTriton1.NativeGraph.reduceValues,TileShape.axisDim,TileShape.insertAxisIndex,hj,hn,
          algebra,SoftmaxTriton1.NativeGraph.algebra,atom,SoftmaxTriton1.NativeGraph.atom,
          masked,SoftmaxTriton1.NativeGraph.masked,Floating.inputRow,numerator]
  | some ms =>
    change p.divFull _ _=row p V n B xs (some ms) i
    unfold row denominator numerator
    congr 1
    · by_cases hi : i<n <;> simp [algebra,SoftmaxTriton1.NativeGraph.algebra,Floating.inputRow,Floating.maskRow,
        masked,SoftmaxTriton1.NativeGraph.masked,hi,atom,SoftmaxTriton1.NativeGraph.atom]
    · change reduceWords p.addRN V B _ _=reduceWords p.addRN V B _ _
      congr 1
      funext j t
      by_cases hj : j<B <;> by_cases hn : j<n <;>
        simp [SoftmaxTriton1.NativeGraph.reduceValues,TileShape.axisDim,TileShape.insertAxisIndex,hj,hn,
          algebra,SoftmaxTriton1.NativeGraph.algebra,atom,SoftmaxTriton1.NativeGraph.atom,
          masked,SoftmaxTriton1.NativeGraph.masked,Floating.inputRow,Floating.maskRow,numerator]

abbrev InstructionAgreement := SoftmaxTriton1.NativeGraph.InstructionAgreement

theorem row_agreement (a b : Instructions) (h : InstructionAgreement a b)
    (V n B : Nat) (xs : Nat → Word) (ms : Option (Nat → Word)) (i : Nat) :
    row a V n B xs ms i=row b V n B xs ms i := by
  rw [SoftmaxTriton1.NativeGraph.instructions_equal a b h]

#specNonCircular row avoiding [Floating.formula,Floating.expression,algebra]
#axiomsClean source_graph_lowering
#axiomsClean row_agreement
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
