import bench.tritonbench_g.l2_norm_triton1.Proofs.Scalar
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.L2NormTriton1.Native
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

abbrev Word := BitVec 32
def word (n : Nat) : Word := BitVec.ofNat 32 n

/-- Exact local PTX ports. NaN payloads, infinities, signed zeros, FTZ and
all overflow outcomes remain part of these raw-word functions. -/
structure Instructions where
  addRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  sqrtApproxFTZ : Word → Word
  divFull : Word → Word → Word

/-- A shuffle stage reads all its operands from the preceding stage. The
self operand is first, including the order seen by each individual lane. -/
def butterfly (p : Instructions) : List Nat → (Nat → Word) → Nat → Word
  | [],values => values
  | distance::rest,values => butterfly p rest
      (fun lane => p.addRN (values lane) (values (Nat.xor lane distance)))

def warpDistances (B : Nat) : List Nat := [16,8,4,2,1].filter (fun d => d<min B 32)
def groupDistances (B : Nat) : List Nat := [2,1].filter (fun d => d<min (B/32) 4)

/-- Default JIT specialization of fresh Torch allocations: aligned pointer
arguments and N divisible by16 permit v2/v4 coalescing; other widths use v1.
This selects the actual compiler register layout, not a numeric identity. -/
def vectorWidth (N B : Nat) : Nat := if N%16=0 then min 4 (max 1 (B/128)) else 1

def registerIndex (V thread slot : Nat) : Nat := (slot/V)*(128*V)+thread*V+slot%V

/-- Register accumulation starts with slot1's rounded square, then fuses
slot0 and slots2.. in order. No FMA/split-arithmetic identity is used. -/
def threadSum (V B : Nat) (leaf : Nat → Nat → Word)
    (fused : Nat → Nat → Word → Word) (thread : Nat) : Word :=
  if B≤128 then leaf thread thread
  else (List.range (B/128-2)).foldl
    (fun accum slot => fused (registerIndex V thread (slot+2)) thread accum)
    (fused (registerIndex V thread 0) thread (leaf (registerIndex V thread 1) thread))

/-- At B=2..128 the first shuffle addition fuses the self square with the
peer's rounded square. Up to one warp, every lane has its own ordered result. Larger
tiles store each warp's lane0 result, reduce the selected2/4 warp results, and
broadcast lane0's result through shared memory. -/
def reduceWords (p : Instructions) (V B : Nat) (leaf : Nat → Nat → Word)
    (fused : Nat → Nat → Word → Word) (lane : Nat) : Word :=
  let warps := if B≤128 then
      match warpDistances B with
      | [] => threadSum V B leaf fused
      | first::rest => butterfly p rest (fun thread =>
          fused thread thread (leaf (Nat.xor thread first) (Nat.xor thread first)))
    else butterfly p (warpDistances B) (threadSum V B leaf fused)
  if B≤32 then warps (lane%max B 1)
  else butterfly p (groupDistances B) (fun warp => warps (warp*32)) 0

/-- Compiler carrier: each scalar retains its per-thread operational value
and the two local operands of a multiply when that node can be fused. This is
source-node provenance, not a numerical law or a successful-output premise. -/
structure Node where
  value : Nat → Word
  product : Option ((Nat → Word) × (Nat → Word)) := none
  singletonSum : Option ((Nat → Word) × (Nat → Word)) := none

instance : Inhabited Node := ⟨⟨fun _ => word 0,none,none⟩⟩
def atom (x : Word) : Node := ⟨fun _ => x,none,none⟩

def multiply (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => p.mulRN (b.value lane) (a.value lane),some (a.value,b.value),none⟩

def fusedNode (p : Instructions) (node : Node) (thread : Nat) (accum : Word) : Word :=
  match node.product with
  | some (a,b) => p.fmaRN (a thread) (b thread) accum
  | none => p.addRN (node.value thread) accum

def reduceNodes (p : Instructions) (V B : Nat) (nodes : Nat → Node) : Node :=
  ⟨reduceWords p V B (fun j t => (nodes j).value t)
      (fun j t accum => fusedNode p (nodes j) t accum),none,
    if B=1 then (nodes 0).product else none⟩

def sumValues (p : Instructions) (V : Nat) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => reduceNodes p V (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨j,h⟩) else atom (word 0))
  · exact fun out => reduceNodes p V (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩) else atom (word 0))

def add (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => match a.singletonSum with
    | some (x,y) => p.fmaRN (x lane) (y lane) (b.value lane)
    | none => p.addRN (b.value lane) (a.value lane),none,none⟩

def algebra (p : Instructions) (V : Nat) : Algebra Node where
  literal := fun _ _ x => atom (if x=1 then word 0x3f800000 else word 0)
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .mul => multiply p a b
    | .add => add p a b
    | .div => ⟨fun lane => p.divFull (a.value lane) (b.value lane),none,none⟩
    | _ => atom (word 0)
  unary := fun _ op a => match op with
    | .sqrt => ⟨fun lane => p.sqrtApproxFTZ (a.value lane),none,none⟩
    | _ => atom (word 0)
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom (word 0)
  fromInt := fun _ _ => atom (word 0)
  fp32Bits := fun _ => atom (word 0)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => atom (word 0)
  reduceSum := fun _ => sumValues p V

def masked (N : Nat) (xs : Nat → Word) (j : Nat) : Word := if j<N then xs j else word 0

/-- Independent instruction graph read from the native reduction schedule.
The guard at j<B is the generic source backend's out-of-shape default; it is
never used by active hardware lanes for any supported power-of-two tile. -/
def squareSum (p : Instructions) (V N B : Nat) (xs : Nat → Word) (lane : Nat) : Word :=
  reduceWords p V B
    (fun j _ => if j<B then p.mulRN (masked N xs j) (masked N xs j) else word 0)
    (fun j _ accum => if j<B then p.fmaRN (masked N xs j) (masked N xs j) accum
      else p.addRN (word 0) accum) lane

def row (p : Instructions) (V N B : Nat) (eps : Word) (xs : Nat → Word) (i : Nat) : Word :=
  let shifted := if B=1 then p.fmaRN (masked N xs 0) (masked N xs 0) eps
    else p.addRN eps (squareSum p V N B xs (i%128))
  p.mulRN (p.divFull (word 0x3f800000) (p.sqrtApproxFTZ shifted)) (xs i)

def squareNodes (p : Instructions) (N B : Nat) (xs : Nat → Word) (j : Nat) : Node :=
  if j<B then multiply p (atom (masked N xs j)) (atom (masked N xs j)) else atom (word 0)

theorem variance_nodes (p : Instructions) (V N B : Nat) (xs : Nat → Word) :
    Tile.variance (algebra p V) N B (fun j => atom (xs j))=
      reduceNodes p V B (squareNodes p N B xs) := by
  unfold Tile.variance
  change reduceNodes p V B _=reduceNodes p V B _
  congr 1
  funext j
  by_cases hj : j<B
  · by_cases active : j<N <;>
      norm_num [squareNodes,hj,TileShape.axisDim,TileShape.insertAxisIndex,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        Tile.masked,Tile.loaded,algebra,masked,active]
  · simp [squareNodes,hj,TileShape.axisDim]


theorem square_nodes_value (p : Instructions) (V N B : Nat) (xs : Nat → Word) (lane : Nat) :
    (reduceNodes p V B (squareNodes p N B xs)).value lane=squareSum p V N B xs lane := by
  have leaf : (fun j t => (squareNodes p N B xs j).value t)=
      (fun j (_ : Nat) => if j<B then p.mulRN (masked N xs j) (masked N xs j) else word 0) := by
    funext j t
    by_cases hj : j<B <;> simp [squareNodes,hj,multiply,atom]
  have fused : (fun j t accum => fusedNode p (squareNodes p N B xs j) t accum)=
      (fun j (_ : Nat) accum => if j<B then p.fmaRN (masked N xs j) (masked N xs j) accum
        else p.addRN (word 0) accum) := by
    funext j t accum
    by_cases hj : j<B <;> simp [squareNodes,hj,fusedNode,multiply,atom]
  change reduceWords p V B _ _ lane=squareSum p V N B xs lane
  rw [leaf,fused]
  rfl


theorem source_graph_lowering (p : Instructions) (V N B : Nat) (eps : Word)
    (xs : Nat → Word) (i : Nat) :
    (Tile.scalar (algebra p V) N B (atom eps) (fun j => atom (xs j)) i).value (i%128)=
      row p V N B eps xs i := by
  unfold Tile.scalar
  rw [variance_nodes]
  have hv := square_nodes_value p V N B xs (i%128)
  by_cases one : B=1
  · subst B
    simp [algebra,multiply,add,reduceNodes,squareNodes,atom,row]
  · simp only [algebra,multiply,add,atom,reduceNodes,one,if_false]
    simpa only [row,one,if_false] using congrArg
      (fun value => p.mulRN (p.divFull (word 0x3f800000)
        (p.sqrtApproxFTZ (p.addRN eps value))) (xs i)) hv

/-- These five optional numerical obligations are exactly local scalar ports.
There is no reduction-output, execution, address, memory, or success field. -/
structure InstructionAgreement (a b : Instructions) : Prop where
  addRN : ∀ x y,a.addRN x y=b.addRN x y
  mulRN : ∀ x y,a.mulRN x y=b.mulRN x y
  fmaRN : ∀ x y z,a.fmaRN x y z=b.fmaRN x y z
  sqrtApproxFTZ : ∀ x,a.sqrtApproxFTZ x=b.sqrtApproxFTZ x
  divFull : ∀ x y,a.divFull x y=b.divFull x y

theorem instructions_equal (a b : Instructions) (h : InstructionAgreement a b) : a=b := by
  cases a;cases b
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun x => funext (h.addRN x),funext fun x => funext (h.mulRN x),
    funext fun x => funext fun y => funext (h.fmaRN x y),funext h.sqrtApproxFTZ,
    funext fun x => funext (h.divFull x)⟩

theorem row_agreement (a b : Instructions) (h : InstructionAgreement a b)
    (V N B : Nat) (eps : Word) (xs : Nat → Word) (i : Nat) :
    row a V N B eps xs i=row b V N B eps xs i := by rw [instructions_equal a b h]

#specNonCircular row avoiding [Tile.scalar,algebra]
#axiomsClean source_graph_lowering
#axiomsClean row_agreement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Native
