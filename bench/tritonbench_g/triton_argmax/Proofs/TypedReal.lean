import bench.tritonbench_g.triton_argmax.Proofs.TypedWhole
import bench.tritonbench_g.triton_argmax.Proofs.Real

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

abbrev Value := Math.Value

def best {n : Nat} (positive : 0<n) (input : Fin n → Value) : Fin n :=
  (Math.enumeration n positive).pick orderedGt orderedEq input

def indexed (dtype : FloatDType) {shape : TileShape} (ax : Fin shape.length)
    (values : Values Value dtype.toTileDType shape) :
    Option (Values Value dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values Value .nat (TileShape.eraseAxis shape ax)) :=
  if positive : 0<TileShape.axisDim shape ax then
    let pick := fun out => best positive (fun lane => toFloat dtype (values (TileShape.insertAxisIndex shape ax out lane)))
    some ((fun out => values (TileShape.insertAxisIndex shape ax out (pick out))),fun out => (pick out).val)
  else none

/-- Same quotation interpreted over mathematical values. Storage casts erase
precision locally; no assertion identifies this with all native dtype results. -/
def backend (rest : Algebra Value) : Algebra Value :=
  { rest with
    negInf := ⊥
    fp32Load := id
    compareEq := fun _ _ => some orderedEq
    compareLt := fun _ _ => some (fun x y => decide (x<y))
    cast := fun _ _ _ x => x
    indexedMax := fun _ dtype => indexed dtype }

def loaded (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (i : Fin BM) (j : Fin BN) : Value :=
  if row+i.val<M ∧ cur+j.val<N then (s.mem X ((row+i.val)*N*K+(cur+j.val)*K+inner)).read .real else ⊥

def next (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Value × Nat) (i : Fin BM) : Value × Nat :=
  let column := cur+(best positive (loaded s X M N K BM BN row inner cur i)).val
  let value : Value := if row+i.val<M ∧ column<N then
    (s.mem X ((row+i.val)*N*K+column*K+inner)).read .real else ⊥
  if (current i).1<value then (value,column) else current i

def vectorLoop (initial : State Value) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (cur : Nat) (current : Fin BM → Value × Nat) : Fin BM → Value × Nat :=
  if h : cur<N then vectorLoop initial X M N K BM BN row inner positive (cur+BN)
    (next initial X M N K BM BN row inner cur positive current) else current
termination_by N-cur
decreasing_by omega

theorem next_memory (initial s : State Value) (same : s.mem=initial.mem) (X : RegionName)
    (M N K BM BN row inner cur : Nat) (positive : 0<BN) (current : Fin BM → Value × Nat) :
    next s X M N K BM BN row inner cur positive current=
      next initial X M N K BM BN row inner cur positive current := by
  funext i
  have loads : loaded s X M N K BM BN row inner cur i=loaded initial X M N K BM BN row inner cur i := by
    funext j
    simp only [loaded,same]
  simp only [next,loads,same]

def written (s : State Value) (O : Region .int) (M K BM row inner : Nat)
    (current : Fin BM → Value × Nat) : State Value :=
  (TileShape.allIndices [BM]).foldl (fun t i =>
    if row+i.1.val<M then t.write (Region.cast O) ((row+i.1.val)*K+inner)
      (.mk .int (Int.ofNat (current i.1).2)) else t) s

theorem written_memory (s initial : State Value) (same : s.mem=initial.mem) (O : Region .int)
    (M K BM row inner : Nat) (current : Fin BM → Value × Nat) :
    (written s O M K BM row inner current).mem=(written initial O M K BM row inner current).mem := by
  simp only [written,Typed.fold_memory,same]

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
