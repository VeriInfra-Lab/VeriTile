import bench.tritonbench_g.decay_cumsum.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.DecayCumsum.Native
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- A reserved SOURCE TEMPLATE slot, never the numerical runtime scale.
The compiler interface binds this occurrence to a raw FP32 parameter. -/
def scaleSlot : ℝ := 2
inductive Origin where
  | raw | loaded | gate | initialZero | invLiteral | scaleParameter
  | forwardProduct | forwardCarry | difference | exponentQ | exponentK
  | scaledQ | qGateProduct | kGateProduct | dq | dk | dqProduct | dkProduct
  | gradient | reverseCarry
  deriving DecidableEq
structure Node where
  value : Word
  origin : Origin := .raw
  factors : Option (Word × Word) := none
instance : Inhabited Node := ⟨⟨positiveZero,.raw,none⟩⟩
def atom (x : Word) : Node := ⟨x,.raw,none⟩
def load (x : Node) : Node := ⟨x.value,.loaded,none⟩
/-- FP32→FP32 casts preserve all bits but reset contraction provenance.
The stored copy does not mutate the original register's provenance. -/
def cast (x : Node) : Node := ⟨x.value,.gate,none⟩
def storedWord (x : Word) : Cell Node := .mk .real ⟨x,.gate,none⟩

def literal (scale : Word) (x : ℝ) : Node :=
  if x=0 then ⟨positiveZero,.initialZero,none⟩
  else if x=1.44269504 then ⟨invLn2,.invLiteral,none⟩
  else if x=scaleSlot then ⟨scale,.scaleParameter,none⟩
  else atom positiveZero

def multiply (p : Instructions) (graph : Graph) (kind : Kind) (a b : Node) : Node :=
  match kind,a.origin,b.origin with
  | .forward,.gate,.invLiteral =>
      ⟨p.mulRN a.value b.value,.forwardProduct,some (a.value,b.value)⟩
  | .prepare,.exponentQ,.scaleParameter => ⟨evalSite p graph .prepareScale a.value b.value,.scaledQ,none⟩
  | .prepare,.loaded,.scaledQ => atom (evalSite p graph .prepareQ a.value b.value)
  | .prepare,.loaded,.exponentK => atom (evalSite p graph .prepareK a.value b.value)
  | .backward,.loaded,.exponentQ =>
      ⟨p.mulRN a.value b.value,.qGateProduct,some (b.value,a.value)⟩
  | .backward,.loaded,.exponentK =>
      ⟨p.mulRN a.value b.value,.kGateProduct,some (b.value,a.value)⟩
  | .backward,.dq,.loaded => ⟨p.mulRN a.value b.value,.dqProduct,some (a.value,b.value)⟩
  | .backward,.dk,.loaded => ⟨evalSite p graph .gradientRight a.value b.value,.dkProduct,some (a.value,b.value)⟩
  | _,_,_ => atom (p.mulRN a.value b.value)

def add (p : Instructions) (graph : Graph) (kind : Kind) (a b : Node) : Node :=
  match kind,a.origin,b.origin,b.factors with
  | .forward,_,.forwardProduct,some (g,_) => ⟨evalSite p graph .forwardStep g a.value,.forwardCarry,none⟩
  | .backward,.loaded,.qGateProduct,some (e,inter) => ⟨evalSite p graph .backwardQ a.value inter e,.dq,none⟩
  | .backward,.loaded,.kGateProduct,some (e,inter) => ⟨evalSite p graph .backwardK a.value inter e,.dk,none⟩
  | .backward,.initialZero,.gradient,_ => ⟨evalSite p graph .firstCarry a.value b.value,.reverseCarry,none⟩
  | .backward,.reverseCarry,.gradient,_ => ⟨evalSite p graph .laterCarry a.value b.value,.reverseCarry,none⟩
  | _,_,_,_ => atom (p.addRN a.value b.value)

def subtract (p : Instructions) (graph : Graph) (kind : Kind) (a b : Node) : Node :=
  match kind,a.origin,b.origin,a.factors with
  | .prepare,.loaded,.gate,_ => ⟨evalSite p graph .prepareDifference a.value b.value,.difference,none⟩
  | .backward,.gate,.gate,_ => ⟨evalSite p graph .backwardDifference a.value b.value,.difference,none⟩
  | .backward,.dqProduct,.dkProduct,some (dq,q) => ⟨evalSite p graph .gradient dq q b.value,.gradient,none⟩
  | _,_,_,_ => atom (p.subRN a.value b.value)
def exponential (p : Instructions) (graph : Graph) (a : Node) : Node :=
  ⟨evalSite p graph (if a.origin=.difference then .exponentK else .exponentQ) a.value positiveZero,
    if a.origin=.difference then .exponentK else .exponentQ,none⟩

def algebra (p : Instructions) (graph : Graph) (kind : Kind) (scale : Word) : Algebra Node where
  literal := fun _ _ => literal scale
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .mul => multiply p graph kind a b
    | .add => add p graph kind a b
    | .sub => subtract p graph kind a b
    | _ => atom positiveZero
  unary := fun _ op a => match op with
    | .exp2 => exponential p graph a
    | _ => atom positiveZero
  cast := fun _ _ _ => cast
  fromNat := fun _ _ => atom positiveZero
  fromInt := fun _ _ => atom positiveZero
  fp32Bits := fun _ => atom positiveZero
  fp32Load := load
  reduceMax := fun _ {_} _ _ _ => fun _ => atom positiveZero
  reduceSum := fun _ {_} _ _ _ => fun _ => atom positiveZero

@[simp] theorem literal_zero (scale : Word) : literal scale 0=⟨positiveZero,.initialZero,none⟩ := by simp [literal]
@[simp] theorem literal_inv (scale : Word) : literal scale 1.44269504=⟨invLn2,.invLiteral,none⟩ := by norm_num [literal]
@[simp] theorem literal_scale (scale : Word) : literal scale scaleSlot=⟨scale,.scaleParameter,none⟩ := by norm_num [literal,scaleSlot]
theorem cast_blocks_product (x : Node) : (cast x).factors=none := rfl
theorem load_blocks_product (x : Node) : (load x).factors=none := rfl
theorem scale_all_words (scale : Word) : (literal scale scaleSlot).value=scale := by simp

#axiomsClean scale_all_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.Native
