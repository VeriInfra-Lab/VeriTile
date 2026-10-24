import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Scalar
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Native

namespace VeriTile.Bench.Repairs.L2NormTriton2.Native
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
abbrev Word := L2NormTriton1.Native.Word
abbrev word := L2NormTriton1.Native.word
abbrev Instructions := L2NormTriton1.Native.Instructions
abbrev reduceWords := L2NormTriton1.Native.reduceWords
abbrev vectorWidth := L2NormTriton1.Native.vectorWidth
abbrev masked := L2NormTriton1.Native.masked
abbrev InstructionAgreement := L2NormTriton1.Native.InstructionAgreement

/-- PTX neg.f32 flips exactly the sign bit, including NaN payloads. -/
def negBits (x : Word) : Word := x ^^^ word 0x80000000

inductive Kind where
  | plain | sum | sqrt | inv | rstd | c1 | c2
  deriving DecidableEq

/-- Compiler provenance selects the observed local instruction lowering.
No equality, finiteness or reduction-output law is a field of this carrier. -/
structure Node where
  value : Nat → Word
  kind : Kind := .plain
  product : Option ((Nat → Word) × (Nat → Word)) := none
  singletonSum : Option ((Nat → Word) × (Nat → Word)) := none

instance : Inhabited Node := ⟨⟨fun _ => word 0,.plain,none,none⟩⟩
def atom (x : Word) : Node := ⟨fun _ => x,.plain,none,none⟩

def multiply (p : Instructions) (B : Nat) (a b : Node) : Node :=
  if a.kind=.sum ∧ b.kind=.inv then
    ⟨fun lane => if B≤32 then p.mulRN (b.value lane) (a.value lane)
      else p.mulRN (a.value lane) (b.value lane),.c1,none,none⟩
  else if a.kind=.c1 ∧ b.kind=.rstd then
    ⟨fun lane => p.mulRN (b.value lane) (a.value lane),.c2,none,none⟩
  else if a.kind=.c2 then
    ⟨fun lane => if B=1 then p.mulRN (a.value lane) (b.value lane)
      else p.mulRN (b.value lane) (a.value lane),.plain,none,none⟩
  else ⟨fun lane => p.mulRN (b.value lane) (a.value lane),.plain,some (b.value,a.value),none⟩

def fusedNode (p : Instructions) (node : Node) (thread : Nat) (accum : Word) : Word :=
  match node.product with
  | some (a,b) => p.fmaRN (a thread) (b thread) accum
  | none => p.addRN (node.value thread) accum

def reduceNodes (p : Instructions) (V B : Nat) (nodes : Nat → Node) : Node :=
  ⟨reduceWords p V B (fun j t => (nodes j).value t)
      (fun j t accum => fusedNode p (nodes j) t accum),.sum,none,
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
    | none => p.addRN (b.value lane) (a.value lane),.plain,none,none⟩

def divide (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => p.divFull (a.value lane) (b.value lane),
    if b.kind=.sqrt then .rstd else .inv,none,none⟩

def subtract (p : Instructions) (a b : Node) : Node :=
  ⟨fun lane => match a.product with
    | some (x,y) => p.fmaRN (x lane) (y lane) (negBits (b.value lane))
    | none => p.addRN (a.value lane) (negBits (b.value lane)),.plain,none,none⟩

def algebra (p : Instructions) (V B : Nat) : Algebra Node where
  literal := fun _ _ x => atom (if x=1 then word 0x3f800000 else word 0)
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .mul => multiply p B a b
    | .add => add p a b
    | .div => divide p a b
    | .sub => subtract p a b
    | _ => atom (word 0)
  unary := fun _ op a => match op with
    | .sqrt => ⟨fun lane => p.sqrtApproxFTZ (a.value lane),.sqrt,none,none⟩
    | _ => atom (word 0)
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom (word 0)
  fromInt := fun _ _ => atom (word 0)
  fp32Bits := fun _ => atom (word 0)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => atom (word 0)
  reduceSum := fun _ => sumValues p V

/-- Independent complete backward native graph, including ordered dot-product
leaves/FMA, B-dependent scale operand order, exact bit negation and final FMA. -/
def productSum (p : Instructions) (V N B : Nat) (xs ys : Nat → Word) (lane : Nat) : Word :=
  reduceWords p V B
    (fun j _ => if j<B then p.mulRN (masked N xs j) (masked N ys j) else word 0)
    (fun j _ accum => if j<B then p.fmaRN (masked N xs j) (masked N ys j) accum
      else p.addRN (word 0) accum) lane

def row (p : Instructions) (V N B : Nat) (eps : Word) (xs dys : Nat → Word) (i : Nat) : Word :=
  let shifted := if B=1 then p.fmaRN (masked N xs 0) (masked N xs 0) eps
    else p.addRN eps (productSum p V N B xs xs (i%128))
  let rstd := p.divFull (word 0x3f800000) (p.sqrtApproxFTZ shifted)
  let inv := p.divFull (word 0x3f800000) shifted
  let dot := productSum p V N B xs dys (i%128)
  let c1 := if B≤32 then p.mulRN inv dot else p.mulRN dot inv
  let c2 := p.mulRN rstd c1
  let c3 := if B=1 then p.mulRN c2 (xs i) else p.mulRN (xs i) c2
  p.fmaRN rstd (dys i) (negBits c3)

def productNodes (p : Instructions) (N B : Nat) (xs ys : Nat → Word) (j : Nat) : Node :=
  if j<B then multiply p B (atom (masked N ys j)) (atom (masked N xs j)) else atom (word 0)

theorem variance_nodes (p : Instructions) (V N B : Nat) (xs : Nat → Word) :
    Tile.variance (algebra p V B) N B (fun j => atom (xs j))=
      reduceNodes p V B (productNodes p N B xs xs) := by
  unfold Tile.variance L2NormTriton1.Tile.variance
  change reduceNodes p V B _=reduceNodes p V B _
  congr 1
  funext j
  by_cases hj : j<B
  · by_cases active : j<N <;>
      norm_num [productNodes,hj,TileShape.axisDim,TileShape.insertAxisIndex,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        L2NormTriton1.Tile.masked,L2NormTriton1.Tile.loaded,algebra,masked,L2NormTriton1.Native.masked,active]
  · simp [productNodes,hj,TileShape.axisDim]

theorem dot_nodes (p : Instructions) (V N B : Nat) (xs dys : Nat → Word) :
    Tile.dot (algebra p V B) N B (fun j => atom (xs j)) (fun j => atom (dys j))=
      reduceNodes p V B (productNodes p N B xs dys) := by
  unfold Tile.dot
  change reduceNodes p V B _=reduceNodes p V B _
  congr 1
  funext j
  by_cases hj : j<B
  · by_cases active : j<N <;>
      norm_num [productNodes,hj,TileShape.axisDim,TileShape.insertAxisIndex,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        Tile.masked,L2NormTriton1.Tile.masked,L2NormTriton1.Tile.loaded,algebra,masked,L2NormTriton1.Native.masked,active]
  · simp [productNodes,hj,TileShape.axisDim]

theorem product_nodes_value (p : Instructions) (V N B : Nat) (xs ys : Nat → Word) (lane : Nat) :
    (reduceNodes p V B (productNodes p N B xs ys)).value lane=productSum p V N B xs ys lane := by
  have leaf : (fun j t => (productNodes p N B xs ys j).value t)=
      (fun j (_ : Nat) => if j<B then p.mulRN (masked N xs j) (masked N ys j) else word 0) := by
    funext j t
    by_cases hj : j<B <;> simp [productNodes,hj,multiply,atom]
  have fused : (fun j t accum => fusedNode p (productNodes p N B xs ys j) t accum)=
      (fun j (_ : Nat) accum => if j<B then p.fmaRN (masked N xs j) (masked N ys j) accum
        else p.addRN (word 0) accum) := by
    funext j t accum
    by_cases hj : j<B <;> simp [productNodes,hj,fusedNode,multiply,atom]
  change reduceWords p V B _ _ lane=productSum p V N B xs ys lane
  rw [leaf,fused]
  rfl

theorem source_graph_lowering (p : Instructions) (V N B : Nat) (eps : Word)
    (xs dys : Nat → Word) (i : Nat) :
    (Tile.scalar (algebra p V B) N B (atom eps) (fun j => atom (xs j))
      (fun j => atom (dys j)) i).value (i%128)=row p V N B eps xs dys i := by
  dsimp only [Tile.scalar]
  rw [variance_nodes,dot_nodes]
  have square := product_nodes_value p V N B xs xs (i%128)
  have cross := product_nodes_value p V N B xs dys (i%128)
  by_cases one : B=1
  · subst B
    simp only [algebra,multiply,add,divide,subtract,atom,reduceNodes,productNodes,
      Kind.noConfusion,reduceIte,↓reduceIte]
    simp_all [row,productNodes,multiply,atom,reduceNodes]
  · dsimp only [reduceNodes] at square cross
    simp [algebra,multiply,add,divide,subtract,atom,reduceNodes,one,row,square,cross]

#specNonCircular row avoiding [Tile.scalar,algebra]
#axiomsClean source_graph_lowering
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Native
