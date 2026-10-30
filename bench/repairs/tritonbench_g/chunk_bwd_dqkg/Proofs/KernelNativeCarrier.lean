import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid

/-! Native FP32/TF32 instructions and compiler routing are local model data.
The selected SM90 family has fixed K8 WGMMA coordinates. Variable-width dots
are explicit chains, reductions cover every padded lane exactly once, and
routing sees only value-erased source provenance. No software equality port. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNativeCarrier
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
abbrev Word := BitVec 32
inductive Columns where
  | n8 | n16 | n32 | n64
  deriving DecidableEq
def Columns.size : Columns → Nat | .n8=>8 | .n16=>16 | .n32=>32 | .n64=>64
structure Instructions where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  exp2Approx : Word → Word
  /-- One coordinate of an SM90 m64nNk8 FP32/TF32 WGMMA. `accumulate=false`
  discards the seed. TF32 and exceptional-value semantics are primitive. -/
  wgmma8 : (columns : Columns) → Fin 64 → Fin columns.size →
    (Fin 8 → Word) → (Fin 8 → Word) → Bool → Word → Word
def negate (x : Word) : Word := x ^^^ BitVec.ofNat 32 0x80000000
inductive Code where
  | input (region : RegionName) (offset : Nat)
  | constant | zero | scale
  | cut (source : Code)
  | add (a b : Code) | sub (a b : Code) | mul (a b : Code)
  | neg (a : Code) | exp (a : Code)
  | dot (D row col : Nat) (a b : Fin D → Code)
inductive AddRoute where
  | separate | right | rightSwapped | left | leftSwapped
  deriving DecidableEq
inductive SubRoute where
  | separate | rightNegLeft | rightNegRight | rightNegLeftSwapped | rightNegRightSwapped
  | leftNegRight | leftNegRightSwapped
  deriving DecidableEq
inductive ReductionTree (n : Nat) where
  | leaf (i : Fin n)
  | zero
  | add (a b : ReductionTree n)
def ReductionTree.leaves {n : Nat} : ReductionTree n → List (Fin n)
  | .leaf i => [i] | .zero => [] | .add a b => a.leaves++b.leaves
structure ReductionPlan (n : Nat) where
  tree : ReductionTree n
  complete : tree.leaves.Perm (List.finRange n)
structure Routing where
  add : Code → AddRoute
  sub : Code → SubRoute
  mulSwap : Code → Bool
  addSwap : Code → Bool
  negSubtractZero : Code → Bool
  dotAccumulate : Code → Bool
  dotInitialize : Code → Bool
  dotColumns : Code → Columns
  reduction : (shape : TileShape) → (ax : Fin shape.length) → (keep : Bool) →
    (TileIndex shape → Code) → TileIndex (TileShape.reduceShape shape ax keep) →
      ReductionPlan (TileShape.axisDim shape ax)

inductive Node where
  | atom (origin : Code) (value : Word)
  | zero | scale (value : Word)
  | add (a b : Node) | sub (a b : Node) | mul (a b : Node)
  | neg (a : Node) | exp (a : Node)
  | dot (D row col : Nat) (a b : Fin D → Node)
instance : Inhabited Node := ⟨.zero⟩
def Node.code : Node → Code
  | .atom origin _ => origin | .zero => .zero | .scale _ => .scale
  | .add a b => .add a.code b.code | .sub a b => .sub a.code b.code | .mul a b => .mul a.code b.code
  | .neg a=>.neg a.code | .exp a=>.exp a.code
  | .dot D row col a b=>.dot D row col (fun i=>(a i).code) (fun i=>(b i).code)

def matrixChain (isa : Instructions) (columns : Columns) (D row col : Nat)
    (a b : Fin D → Word) (accumulate : Bool) (seed : Word) : Word :=
  ((List.range ((D+7)/8)).foldl (fun state block =>
    (Bool.true,isa.wgmma8 columns ⟨row%64,Nat.mod_lt _ (by decide)⟩
      ⟨col%columns.size,Nat.mod_lt _ (by cases columns <;> decide)⟩
      (fun j=>if h : block*8+j.val<D then a ⟨block*8+j.val,h⟩ else 0)
      (fun j=>if h : block*8+j.val<D then b ⟨block*8+j.val,h⟩ else 0)
      state.1 state.2)) (accumulate,seed)).2
def addValue (isa : Instructions) (routing : Routing) (code : Code) (a b : Word) : Word :=
  if routing.addSwap code then isa.addRN b a else isa.addRN a b
def Node.isZero : Node → Bool | .zero=>Bool.true | _=>Bool.false
structure Evaluation where
  value : Word
  product : Option (Word × Word) := none
  matrix : Option (Bool → Word → Word) := none

def fusedAdd (isa : Instructions) (route : AddRoute) (left right : Evaluation) (fallback : Word) : Word :=
  match route,left.product,right.product with
  | .left,some (a,b),_=>isa.fmaRN a b right.value
  | .leftSwapped,some (a,b),_=>isa.fmaRN b a right.value
  | .right,_,some (a,b)=>isa.fmaRN a b left.value
  | .rightSwapped,_,some (a,b)=>isa.fmaRN b a left.value
  | _,_,_=>fallback

def fusedSub (isa : Instructions) (route : SubRoute) (left right : Evaluation) : Word :=
  match route,left.product,right.product with
  | .leftNegRight,some (a,b),_=>isa.fmaRN a b (negate right.value)
  | .leftNegRightSwapped,some (a,b),_=>isa.fmaRN b a (negate right.value)
  | .rightNegLeft,_,some (a,b)=>isa.fmaRN (negate a) b left.value
  | .rightNegRight,_,some (a,b)=>isa.fmaRN a (negate b) left.value
  | .rightNegLeftSwapped,_,some (a,b)=>isa.fmaRN b (negate a) left.value
  | .rightNegRightSwapped,_,some (a,b)=>isa.fmaRN (negate b) a left.value
  | _,_,_=>isa.subRN left.value right.value

/-- Product and matrix fields retain only local already evaluated operands,
allowing instruction contraction without recursive grandchild evaluation. -/
def Node.evaluate (isa : Instructions) (routing : Routing) : Node → Evaluation
  | .atom _ value=>⟨value,none,none⟩
  | .zero=>⟨0,none,none⟩
  | .scale value=>⟨value,none,none⟩
  | .mul a b=>
      let x:=(a.evaluate isa routing).value
      let y:=(b.evaluate isa routing).value
      ⟨if routing.mulSwap (.mul a.code b.code) then isa.mulRN y x else isa.mulRN x y,some (x,y),none⟩
  | .neg a=>⟨if routing.negSubtractZero (.neg a.code) then isa.subRN 0 (a.evaluate isa routing).value
      else negate (a.evaluate isa routing).value,none,none⟩
  | .exp a=>⟨isa.exp2Approx (isa.mulRN (a.evaluate isa routing).value 0x3fb8aa3b),none,none⟩
  | .dot D row col a b=>
      let chain:=matrixChain isa (routing.dotColumns (.dot D row col (fun j=>(a j).code) (fun j=>(b j).code)))
        D row col (fun j=>((a j).evaluate isa routing).value) (fun j=>((b j).evaluate isa routing).value)
      ⟨chain Bool.false 0,none,some chain⟩
  | .add a b=>
      let left:=a.evaluate isa routing
      let right:=b.evaluate isa routing
      let code:=Code.add a.code b.code
      let fallback:=addValue isa routing code left.value right.value
      let value:=match right.matrix with
        | some chain=>if routing.dotAccumulate code then chain Bool.true left.value
            else if routing.dotInitialize code && a.isZero then chain Bool.false 0 else fallback
        | none=>fusedAdd isa (routing.add code) left right fallback
      ⟨value,none,none⟩
  | .sub a b=>⟨fusedSub isa (routing.sub (.sub a.code b.code))
      (a.evaluate isa routing) (b.evaluate isa routing),none,none⟩

def Node.eval (isa : Instructions) (routing : Routing) (x : Node) : Word := (x.evaluate isa routing).value

def materialize (isa : Instructions) (routing : Routing) (value : Node) : Node :=
  .atom (.cut value.code) (value.eval isa routing)
def ReductionTree.expression {n : Nat} (values : Fin n → Node) : ReductionTree n → Node
  | .leaf i => values i | .zero => .zero | .add a b => .add (a.expression values) (b.expression values)
def reduce (routing : Routing) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (values : Values Node .real shape) : Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ((routing.reduction shape ax Bool.false (fun i => (values i).code) out).tree).expression
      (fun i => values (TileShape.insertAxisIndex shape ax out i))
  · exact fun out => ((routing.reduction shape ax Bool.true (fun i => (values i).code) out).tree).expression
      (fun i => values (TileShape.replaceAxisIndex shape ax out i))

/-- Runtime scale occupies a distinguished source literal slot, preserving -0.
The selected quote contains only that slot and literal zero. Compatibility
casts are FP32-to-FP32 identities; memory loads materialize a word. -/
def scaleSlot : ℝ := -1
def nativeConfig (c : Whole.Config) : Whole.Config := {c with scale:=scaleSlot}
def algebra (isa : Instructions) (routing : Routing) (scale : Word) : Algebra Node where
  literal := fun _ _ x => if x=scaleSlot then .scale scale else .zero
  negInf := .atom .constant 0xff800000
  binary := fun _ _ op a b => match op with | .add => .add a b | .sub => .sub a b | .mul => .mul a b | _ => .zero
  unary := fun _ op a => match op with | .neg _=>.neg a | .exp=>.exp a | _=>.zero
  cast := fun _ _ _ a=>a
  fromNat := fun _ _=>.zero
  fromInt := fun _ _=>.zero
  fp32Bits := fun x=>.atom .constant x.bits
  fp32Load := materialize isa routing
  reduceMax := fun _ {_} _ _ _ _=>.zero
  reduceSum := fun _ {_}=>reduce routing

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Node left.toTileDType (batch++[m,k]) →
      Values Node right.toTileDType (batch++[k,n]) →
      Values Node out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>FP.Structural.ofFloat out (.dot D i.1.val i.2.1.val
      (fun d=>FP.Structural.toFloat left (a (i.1,d,())))
      (fun d=>FP.Structural.toFloat right (b (d,i.2.1,()))))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds
      (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2
def dotModel : Typed.Dot Node where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues p l r out options batch a b

def observe (isa : Instructions) (routing : Routing) : Cell Node → Word
  | .mk .real x | .mk .fp32 x => x.eval isa routing
  | _=>0

theorem matrix_coverage (D : Nat) (j : Fin D) : j.val/8<(D+7)/8 ∧ j.val/8*8+j.val%8=j.val := by
  have := j.isLt
  omega

theorem negative_zero_scale (isa : Instructions) (routing : Routing) :
    ((algebra isa routing 0x80000000).literal (some .fp32) .real scaleSlot).eval isa routing=0x80000000 := by
  simp [algebra,Node.eval,Node.evaluate]

#axiomsClean matrix_coverage
#axiomsClean negative_zero_scale
#specNonCircular algebra avoiding [Typed.runGrid,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNativeCarrier
