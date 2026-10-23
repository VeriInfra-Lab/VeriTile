import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardReference
/-! Native scalar instructions and compiler routing are model data. Routing
inspects value-erased provenance only. Every reduction contains each padded
lane exactly once. Casts materialize FP32 words and stop contraction across
source stores and reloads. No primitive=software equality is assumed. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeCarrier
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
abbrev Word := BitVec 32
structure Instructions where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
def negate (x : Word) : Word := x ^^^ BitVec.ofNat 32 0x80000000
inductive Code where
  | input (region : RegionName) (offset : Nat)
  | constant | zero | scale
  | cut (source : Code)
  | add (a b : Code) | sub (a b : Code) | mul (a b : Code)
inductive AddRoute where
  | separate | right | rightSwapped | left | leftSwapped
  deriving DecidableEq
inductive SubRoute where
  | separate | rightNegLeft | rightNegRight | rightNegLeftSwapped | rightNegRightSwapped
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
  reduction : (shape : TileShape) → (ax : Fin shape.length) → (keep : Bool) →
    (TileIndex shape → Code) → TileIndex (TileShape.reduceShape shape ax keep) →
      ReductionPlan (TileShape.axisDim shape ax)

inductive Node where
  | atom (origin : Code) (value : Word)
  | zero | scale (value : Word)
  | add (a b : Node) | sub (a b : Node) | mul (a b : Node)
instance : Inhabited Node := ⟨.zero⟩
def Node.code : Node → Code
  | .atom origin _ => origin | .zero => .zero | .scale _ => .scale
  | .add a b => .add a.code b.code | .sub a b => .sub a.code b.code | .mul a b => .mul a.code b.code
def Node.eval (isa : Instructions) (routing : Routing) : Node → Word
  | .atom _ value => value | .zero => 0 | .scale value => value
  | .mul a b => if routing.mulSwap (.mul a.code b.code) then isa.mulRN (b.eval isa routing) (a.eval isa routing)
      else isa.mulRN (a.eval isa routing) (b.eval isa routing)
  | .add a b => match routing.add (.add a.code b.code) with
    | .right => match hb : b with
      | .mul x y => isa.fmaRN (x.eval isa routing) (y.eval isa routing) (a.eval isa routing)
      | _ => isa.addRN (a.eval isa routing) (b.eval isa routing)
    | .rightSwapped => match hb : b with
      | .mul x y => isa.fmaRN (y.eval isa routing) (x.eval isa routing) (a.eval isa routing)
      | _ => isa.addRN (a.eval isa routing) (b.eval isa routing)
    | .left => match ha : a with
      | .mul x y => isa.fmaRN (x.eval isa routing) (y.eval isa routing) (b.eval isa routing)
      | _ => isa.addRN (a.eval isa routing) (b.eval isa routing)
    | .leftSwapped => match ha : a with
      | .mul x y => isa.fmaRN (y.eval isa routing) (x.eval isa routing) (b.eval isa routing)
      | _ => isa.addRN (a.eval isa routing) (b.eval isa routing)
    | .separate => isa.addRN (a.eval isa routing) (b.eval isa routing)
  | .sub a b => match routing.sub (.sub a.code b.code) with
    | .rightNegLeft => match hb : b with
      | .mul x y => isa.fmaRN (negate (x.eval isa routing)) (y.eval isa routing) (a.eval isa routing)
      | _ => isa.subRN (a.eval isa routing) (b.eval isa routing)
    | .rightNegRight => match hb : b with
      | .mul x y => isa.fmaRN (x.eval isa routing) (negate (y.eval isa routing)) (a.eval isa routing)
      | _ => isa.subRN (a.eval isa routing) (b.eval isa routing)
    | .rightNegLeftSwapped => match hb : b with
      | .mul x y => isa.fmaRN (y.eval isa routing) (negate (x.eval isa routing)) (a.eval isa routing)
      | _ => isa.subRN (a.eval isa routing) (b.eval isa routing)
    | .rightNegRightSwapped => match hb : b with
      | .mul x y => isa.fmaRN (negate (y.eval isa routing)) (x.eval isa routing) (a.eval isa routing)
      | _ => isa.subRN (a.eval isa routing) (b.eval isa routing)
    | .separate => isa.subRN (a.eval isa routing) (b.eval isa routing)
termination_by e => sizeOf e
decreasing_by all_goals simp_all <;> omega

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

def scaleSlot : ℝ := -1
def algebra (isa : Instructions) (routing : Routing) (scale : Word) : Algebra Node where
  literal := fun _ _ x => if x=scaleSlot then .scale scale else .zero
  negInf := .atom .constant (BitVec.ofNat 32 0xff800000)
  binary := fun _ _ op a b => match op with | .add => .add a b | .sub => .sub a b | .mul => .mul a b | _ => .zero
  unary := fun _ _ _ => .zero
  cast := fun _ _ _ => materialize isa routing
  fromNat := fun _ _ => .zero
  fromInt := fun _ _ => .zero
  fp32Bits := fun x => .atom .constant x.bits
  fp32Load := materialize isa routing
  reduceMax := fun _ {_} _ _ _ _ => .zero
  reduceSum := fun _ {_} => reduce routing
def observe (isa : Instructions) (routing : Routing) (cell : Cell Node) : Word :=
  (cell.read .real).eval isa routing
def inputCell (r : RegionName) (off : Nat) (word : Word) : Cell Node := .mk .real (.atom (.input r off) word)

theorem stored_observation (isa : Instructions) (routing : Routing) (scale : Word) (value : Node) :
    observe isa routing (KernelReference.stored (algebra isa routing scale) value)=value.eval isa routing := by
  simp [observe,KernelReference.stored,KernelReference.cast,algebra,materialize,Node.eval]
theorem input_observation (isa : Instructions) (routing : Routing) (r : RegionName) (off : Nat) (word : Word) :
    observe isa routing (inputCell r off word)=word := by
  simp [observe,inputCell,Node.eval]

#axiomsClean stored_observation
#axiomsClean input_observation
#specNonCircular algebra avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNativeCarrier
