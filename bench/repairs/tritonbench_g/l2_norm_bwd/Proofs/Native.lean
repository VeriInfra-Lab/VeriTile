import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Physical
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.L2NormBwd.Native
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

abbrev Word := BitVec 32
def word (n : Nat) : Word := BitVec.ofNat 32 n

/-- Actual PTX primitive ports, over every raw binary32 pattern. In particular
sqrt.approx.ftz and div.full are distinct; no rsqrt substitution is made.
neg.f32 remains explicit, including any exceptional-value behavior. -/
structure Instructions where
  addRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  sqrtApproxFTZ : Word → Word
  divFull : Word → Word → Word
  negate : Word → Word

def butterfly (p : Instructions) : List Nat → (Nat → Word) → Nat → Word
  | [],values => values
  | distance::rest,values => butterfly p rest
      (fun lane => p.addRN (values lane) (values (Nat.xor lane distance)))

def warpDistances (B : Nat) := [16,8,4,2,1].filter (fun d => d<min B 32)
def groupDistances (B : Nat) := [2,1].filter (fun d => d<min (B/32) 4)

def threadSum (B : Nat) (leaf : Nat → Nat → Word)
    (fused : Nat → Nat → Word → Word) (thread : Nat) : Word :=
  if B≤128 then leaf thread thread
  else (List.range (B/128-2)).foldl
    (fun accum slot => fused (thread+128*(slot+2)) thread accum)
    (fused thread thread (leaf (thread+128) thread))

/-- Every shuffle reads the preceding stage; all operand orders are retained.
At B=2..128 the first shuffle fuses self product with peer rounded product.
Above128 each thread starts slot1 rounded, then FMA slot0 and slots2.. . -/
def reduceWords (p : Instructions) (B : Nat) (leaf : Nat → Nat → Word)
    (fused : Nat → Nat → Word → Word) (lane : Nat) : Word :=
  let warps := if B≤128 then
      match warpDistances B with
      | [] => threadSum B leaf fused
      | first::rest => butterfly p rest (fun thread =>
          fused thread thread (leaf (Nat.xor thread first) (Nat.xor thread first)))
    else butterfly p (warpDistances B) (threadSum B leaf fused)
  if B≤32 then warps (lane%max B 1)
  else butterfly p (groupDistances B) (fun warp => warps (warp*32)) 0

/-- Operation-node provenance represents the compiler's local fusion choices;
it contains no predicate about an output, successful execution or memory. -/
structure Node where
  value : Nat → Word
  product : Option ((Nat → Word) × (Nat → Word)) := none
  singletonSum : Option ((Nat → Word) × (Nat → Word)) := none
  isSum : Bool := Bool.false

instance : Inhabited Node := ⟨⟨fun _ => word 0,none,none,Bool.false⟩⟩
def atom (x : Word) : Node := ⟨fun _ => x,none,none,Bool.false⟩

def multiply (p : Instructions) (B : Nat) (a b : Node) : Node :=
  ⟨fun lane => if a.isSum && decide (32<B) then p.mulRN (a.value lane) (b.value lane)
    else p.mulRN (b.value lane) (a.value lane),some (a.value,b.value),none,Bool.false⟩

def fusedNode (p : Instructions) (node : Node) (thread : Nat) (accum : Word) : Word :=
  match node.product with
  | some (a,b) => p.fmaRN (b thread) (a thread) accum
  | none => p.addRN (node.value thread) accum

def reduceNodes (p : Instructions) (B : Nat) (nodes : Nat → Node) : Node :=
  ⟨reduceWords p B (fun j t => (nodes j).value t)
      (fun j t accum => fusedNode p (nodes j) t accum),none,
    if B=1 then (nodes 0).product else none,Bool.true⟩

def sumValues (p : Instructions) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => reduceNodes p (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨j,h⟩) else atom (word 0))
  · exact fun out => reduceNodes p (TileShape.axisDim shape ax) (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩) else atom (word 0))

def add (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => match a.singletonSum with
    | some (x,y) => p.fmaRN (y lane) (x lane) (b.value lane)
    | none => p.addRN (b.value lane) (a.value lane),none,none,Bool.false⟩

def subtract (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => match a.product with
    | some (x,y) => p.fmaRN (y lane) (x lane) (p.negate (b.value lane))
    | none => p.addRN (a.value lane) (p.negate (b.value lane)),none,none,Bool.false⟩

def algebra (p : Instructions) (B : Nat) : Algebra Node where
  literal := fun _ _ x => atom (if x=1 then word 0x3f800000 else word 0)
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .mul => multiply p B a b
    | .add => add p a b
    | .sub => subtract p a b
    | .div => ⟨fun lane => p.divFull (a.value lane) (b.value lane),none,none,Bool.false⟩
    | _ => atom (word 0)
  unary := fun _ op a => match op with
    | .sqrt => ⟨fun lane => p.sqrtApproxFTZ (a.value lane),none,none,Bool.false⟩
    | _ => atom (word 0)
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom (word 0)
  fromInt := fun _ _ => atom (word 0)
  fp32Bits := fun _ => atom (word 0)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => atom (word 0)
  reduceSum := fun _ => sumValues p

def masked (N : Nat) (xs : Nat → Word) (j : Nat) : Word := if j<N then xs j else word 0

def productReduction (p : Instructions) (N B : Nat) (xs ys : Nat → Word) (lane : Nat) : Word :=
  reduceWords p B
    (fun j _ => if j<B then p.mulRN (masked N ys j) (masked N xs j) else word 0)
    (fun j _ accum => if j<B then p.fmaRN (masked N ys j) (masked N xs j) accum
      else p.addRN (word 0) accum) lane

/-- Independently written raw instruction graph, including the B-dependent
dot-times-reciprocal operand order and the final neg.f32/FMA pair. -/
def row (p : Instructions) (N B : Nat) (eps : Word) (xs dys : Nat → Word) (i : Nat) : Word :=
  let q := if B=1 then p.fmaRN (masked N xs 0) (masked N xs 0) eps
    else p.addRN eps (productReduction p N B xs xs (i%128))
  let rs := p.divFull (word 0x3f800000) (p.sqrtApproxFTZ q)
  let inv := p.divFull (word 0x3f800000) q
  let dot := productReduction p N B dys xs (i%128)
  let scaled := if 32<B then p.mulRN dot inv else p.mulRN inv dot
  let correction := p.mulRN (xs i) (p.mulRN rs scaled)
  p.fmaRN rs (dys i) (p.negate correction)

def productNodes (p : Instructions) (N B : Nat) (xs ys : Nat → Word) (j : Nat) : Node :=
  if j<B then multiply p B (atom (masked N xs j)) (atom (masked N ys j)) else atom (word 0)

theorem product_nodes (p : Instructions) (N B : Nat) (xs ys : Nat → Word) :
    Tile.productSum (algebra p B) N B (fun j => atom (xs j)) (fun j => atom (ys j))=
      reduceNodes p B (productNodes p N B xs ys) := by
  unfold Tile.productSum
  change reduceNodes p B _=reduceNodes p B _
  congr 1
  funext j
  by_cases hj : j<B
  · by_cases active : j<N <;>
      norm_num [productNodes,hj,TileShape.axisDim,TileShape.insertAxisIndex,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        Tile.masked,Tile.loaded,algebra,masked,active]
  · simp [productNodes,hj,TileShape.axisDim]

theorem product_nodes_value (p : Instructions) (N B : Nat) (xs ys : Nat → Word) (lane : Nat) :
    (reduceNodes p B (productNodes p N B xs ys)).value lane=productReduction p N B xs ys lane := by
  have leaf : (fun j t => (productNodes p N B xs ys j).value t)=
      (fun j (_ : Nat) => if j<B then p.mulRN (masked N ys j) (masked N xs j) else word 0) := by
    funext j t
    by_cases hj : j<B <;> simp [productNodes,hj,multiply,atom]
  have fused : (fun j t accum => fusedNode p (productNodes p N B xs ys j) t accum)=
      (fun j (_ : Nat) accum => if j<B then p.fmaRN (masked N ys j) (masked N xs j) accum
        else p.addRN (word 0) accum) := by
    funext j t accum
    by_cases hj : j<B <;> simp [productNodes,hj,fusedNode,multiply,atom]
  change reduceWords p B _ _ lane=productReduction p N B xs ys lane
  rw [leaf,fused]
  rfl

theorem source_graph_lowering (p : Instructions) (N B : Nat) (eps : Word)
    (xs dys : Nat → Word) (i : Nat) :
    (Tile.scalar (algebra p B) N B (atom eps) (fun j => atom (xs j)) (fun j => atom (dys j)) i).value (i%128)=
      row p N B eps xs dys i := by
  unfold Tile.scalar
  rw [product_nodes,product_nodes]
  by_cases one : B=1
  · subst B
    simp [algebra,multiply,add,subtract,reduceNodes,productNodes,atom,row,
      productReduction,reduceWords,warpDistances,groupDistances,threadSum,Nat.mod_one]
  · have hv := product_nodes_value p N B xs xs (i%128)
    have hd := product_nodes_value p N B dys xs (i%128)
    simp only [algebra,multiply,add,subtract,atom,reduceNodes,one,if_false]
    by_cases big : 32<B <;> simp [row,one,big,←hv,←hd] <;> rfl

/-- Raw memory uses actual 32-bit words; the auxiliary provenance nodes are
confined to the constructive source-lowering theorem. -/
def reduction (p : Instructions) : Physical.Reduction Word where
  apply := row p
  congr := by
    intro N B eps xs dys us vs hx hy i hi
    rw [←source_graph_lowering p N B eps xs dys i,←source_graph_lowering p N B eps us vs i]
    apply congrArg (fun n : Node => n.value (i%128))
    exact Tile.scalar_congr (algebra p B) N B (atom eps) _ _ _ _
      (fun j hj => congrArg atom (hx j hj)) (fun j hj => congrArg atom (hy j hj)) i hi

/-- Optional primitive model binding between two instruction interpretations.
These are semantic boundaries, not deferred numerical experiments. The program,
reduction schedule, output mapping and memory frame are not fields. -/
structure Obligations (actual target : Instructions) : Prop where
  add : ∀ a b,actual.addRN a b=target.addRN a b
  mul : ∀ a b,actual.mulRN a b=target.mulRN a b
  fma : ∀ a b c,actual.fmaRN a b c=target.fmaRN a b c
  sqrt : ∀ a,actual.sqrtApproxFTZ a=target.sqrtApproxFTZ a
  div : ∀ a b,actual.divFull a b=target.divFull a b
  neg : ∀ a,actual.negate a=target.negate a

theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) : actual=target := by
  cases actual;cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun a => funext (h.add a),funext fun a => funext (h.mul a),
    funext fun a => funext fun b => funext (h.fma a b),funext h.sqrt,
    funext fun a => funext (h.div a),funext h.neg⟩

theorem row_equiv (actual target : Instructions) (h : Obligations actual target)
    (N B : Nat) (eps : Word) (xs dys : Nat → Word) (i : Nat) :
    row actual N B eps xs dys i=row target N B eps xs dys i := by
  rw [instructions_eq actual target h]

#specNonCircular row avoiding [algebra,Tile.scalar,source]
#axiomsClean source_graph_lowering
#axiomsClean row_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Native
