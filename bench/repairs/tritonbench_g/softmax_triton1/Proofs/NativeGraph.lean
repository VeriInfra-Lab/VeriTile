import bench.repairs.tritonbench_g.softmax_triton1.Proofs.FPKernel
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Native

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Word := BitVec 32
abbrev Node := Nat → Word
def atom (x : Word) : Node := fun _ => x

def negInf : Word := 0xff800000
def log2e : Word := 0x3fb8aa3b

/-- Scalar native instruction semantics at FP32 raw-word precision. These
primitive ports are a modeling boundary, not per-task numerical experiments. -/
structure Instructions where
  maxNum : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  exp2Approx : Word → Word
  addRN : Word → Word → Word
  divFull : Word → Word → Word

/-- Reuse the reviewed simultaneous self-first butterfly combinator. The
other L2 fields are unused; no L2 square/FMA schedule is imported. -/
def reductionPorts (op : Word → Word → Word) :
    VeriTile.Bench.Repairs.L2NormTriton1.Native.Instructions :=
  ⟨op,fun _ _ => 0,fun _ _ _ => 0,fun _ => 0,fun _ _ => 0⟩

def butterfly (op : Word → Word → Word) :=
  VeriTile.Bench.Repairs.L2NormTriton1.Native.butterfly (reductionPorts op)

/-- V is the actual sizePerThread from the compiled blocked layout. The
wrapper's width-dependent warp count is retained for every B. -/
structure Profile where
  vector : Nat
  allowed : vector=1 ∨ vector=2 ∨ vector=4

def threads (B : Nat) := 32*numWarps B

def registerIndex (V B thread slot : Nat) : Nat :=
  (slot/V)*(threads B*V)+thread*V+slot%V

def owner (V B i : Nat) : Nat := (i/V)%threads B

def warpDistances (V B : Nat) : List Nat :=
  [16,8,4,2,1].filter (fun d => d < min (B/V) 32)

def groupDistances (V B : Nat) : List Nat :=
  [8,4,2,1].filter (fun d => d < min (B/(32*V)) (numWarps B))

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
      TileShape.insertAxisIndex,algebra,atom,masked]
  unfold Floating.formula
  rw [dif_pos active]
  unfold Floating.rowExpression
  rw [maxeq]
  change p.divFull _ _=row p V n B xs i
  unfold row denominator numerator
  congr 1
  · by_cases hi : i<n <;> simp [algebra,masked,hi,atom]
  · change reduceWords p.addRN V B _ _=reduceWords p.addRN V B _ _
    congr 1
    funext j t
    by_cases hj : j<B
    · by_cases hn : j<n <;>
        simp [reduceValues,TileShape.axisDim,TileShape.insertAxisIndex,hj,hn,
          algebra,atom,masked,numerator]
    · simp [reduceValues,TileShape.axisDim,hj,atom]

/-- Optional transport between two instruction models. Primitive instruction
conformance is not classified as a softmax transformation experiment. -/
structure InstructionAgreement (a b : Instructions) : Prop where
  maxNum : ∀ x y,a.maxNum x y=b.maxNum x y
  subRN : ∀ x y,a.subRN x y=b.subRN x y
  mulRN : ∀ x y,a.mulRN x y=b.mulRN x y
  exp2Approx : ∀ x,a.exp2Approx x=b.exp2Approx x
  addRN : ∀ x y,a.addRN x y=b.addRN x y
  divFull : ∀ x y,a.divFull x y=b.divFull x y

theorem instructions_equal (a b : Instructions) (h : InstructionAgreement a b) : a=b := by
  cases a;cases b
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun x => funext (h.maxNum x),funext fun x => funext (h.subRN x),
    funext fun x => funext (h.mulRN x),funext h.exp2Approx,
    funext fun x => funext (h.addRN x),funext fun x => funext (h.divFull x)⟩

theorem row_agreement (a b : Instructions) (h : InstructionAgreement a b)
    (V n B : Nat) (xs : Nat → Word) (i : Nat) :
    row a V n B xs i=row b V n B xs i := by rw [instructions_equal a b h]

#specNonCircular row avoiding [Floating.formula,Floating.rowExpression,algebra]
#axiomsClean source_graph_lowering
#axiomsClean row_agreement
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.NativeGraph
