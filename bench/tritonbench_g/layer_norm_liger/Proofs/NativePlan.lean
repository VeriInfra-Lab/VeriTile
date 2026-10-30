import bench.tritonbench_g.layer_norm_liger.Proofs.GridHost
namespace VeriTile.Bench.Repairs.LayerNormLiger.NativePlan
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- A scalar expression keeps multiplication operands until a permitted local
contraction is selected. Stores/casts form barriers; no whole row or kernel
result can be supplied as a primitive. -/
inductive Expr where
  | word (value : FP.Value .fp32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Expr)
  | fma (precision : Option ComputeDType) (dtype : FloatDType) (x y z : Expr)
  | neg (value : Expr)
  | rsqrt (precision : Option ComputeDType) (value : Expr)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Expr)
  | hostAdd (left right : Expr)
  | hostCast (value : Expr)

instance : Inhabited Expr := ⟨.word (FP.Value.zero .fp32)⟩

/-- Each decision selects just one syntactically present mul/add or mul/sub.
It never asserts that a fused result equals a separately rounded result. -/
abbrev Contraction := Option ComputeDType → FloatDType → Binary → Expr → Expr → Bool

def binary (contract : Contraction) (pr : Option ComputeDType) (d : FloatDType)
    (op : Binary) (x y : Expr) : Expr :=
  if contract pr d op x y then
    match op,x,y with
    | .add,.binary _ _ .mul a b,z => .fma pr d a b z
    | .add,z,.binary _ _ .mul a b => .fma pr d a b z
    | .sub,.binary _ _ .mul a b,z => .fma pr d a b (.neg z)
    | .sub,z,.binary _ _ .mul a b => .fma pr d (.neg a) b z
    | _,_,_ => .binary pr d op x y
  else .binary pr d op x y

structure Primitives where
  scalarConfig : Option ComputeDType → FloatDType → Binary → FP.Config
  fmaConfig : Option ComputeDType → FloatDType → FP.Config
  hostConfig : FP.Config
  literalFP32 : Option ComputeDType → FloatDType → ℝ → FP.Value .fp32
  rsqrtFP32 : Option ComputeDType → FP.Value .fp32 → FP.Value .fp32
  castFP32 : Option ComputeDType → FloatDType → FloatDType → FP.Value .fp32 → FP.Value .fp32
  hostCastFP32 : FP.Value .fp32 → FP.Value .fp32

/-- All arithmetic is explicit scalar software arithmetic, including actual
one-rounding FMA. Literal/cast/rsqrt numerical correspondence stays local. -/
def Expr.eval (P : Primitives) : Expr → FP.Value .fp32
  | .word v => v
  | .literal pr d v => P.literalFP32 pr d v
  | .binary pr d op x y => match op with
      | .add => FP.add (P.scalarConfig pr d .add) (x.eval P) (y.eval P)
      | .sub => FP.sub (P.scalarConfig pr d .sub) (x.eval P) (y.eval P)
      | .mul => FP.mul (P.scalarConfig pr d .mul) (x.eval P) (y.eval P)
      | .div => FP.div (P.scalarConfig pr d .div) (x.eval P) (y.eval P)
      | .max | .pow => FP.Value.qNaN .fp32
  | .fma pr d x y z => FP.fma (P.fmaConfig pr d) (x.eval P) (y.eval P) (z.eval P)
  | .neg x => (x.eval P).neg
  | .rsqrt pr x => P.rsqrtFP32 pr (x.eval P)
  | .cast pr src dst x => P.castFP32 pr src dst (x.eval P)
  | .hostAdd x y => FP.add P.hostConfig (x.eval P) (y.eval P)
  | .hostCast x => P.hostCastFP32 (x.eval P)

def realBinary (op : Binary) (x y : ℝ) : ℝ := match op with
  | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
  | .max => max x y | .pow => Real.rpow x y

def Expr.ideal (input : FP.Value .fp32 → ℝ) : Expr → ℝ
  | .word v => input v
  | .literal _ _ v => v
  | .binary _ _ op x y => realBinary op (x.ideal input) (y.ideal input)
  | .fma _ _ x y z => x.ideal input*y.ideal input+z.ideal input
  | .neg x => -(x.ideal input)
  | .rsqrt _ x => 1 / Real.sqrt (x.ideal input)
  | .cast _ _ _ x => x.ideal input
  | .hostAdd x y => x.ideal input+y.ideal input
  | .hostCast x => x.ideal input

/-- These are local real identities, not bit equality of fused/separate paths. -/
theorem fma_add_ideal (input : FP.Value .fp32 → ℝ) (pr : Option ComputeDType)
    (d : FloatDType) (x y z : Expr) :
    (Expr.fma pr d x y z).ideal input=
      (Expr.binary pr d .add (.binary pr d .mul x y) z).ideal input := rfl

theorem fma_sub_ideal (input : FP.Value .fp32 → ℝ) (pr : Option ComputeDType)
    (d : FloatDType) (x y z : Expr) :
    (Expr.fma pr d (.neg x) y z).ideal input=
      (Expr.binary pr d .sub z (.binary pr d .mul x y)).ideal input := by
  simp [Expr.ideal,realBinary]
  ring

/-- A source reduction receives an explicit complete leaf tree. Selection may
vary by operation/shape; the plan cannot omit, duplicate or replace leaves. -/
abbrev Trees := Option ComputeDType → (n : Nat) → (Fin n → Expr) → ReductionPlan.Plan n

def sum (contract : Contraction) (trees : Trees) (pr : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape) :
    Values Expr .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out =>
      let leaves := fun i => xs (TileShape.insertAxisIndex shape ax out i)
      (trees pr (TileShape.axisDim shape ax) leaves).tree.eval (.literal pr .fp32 0)
        (binary contract pr .fp32 .add) leaves
  · exact fun out =>
      let leaves := fun i => xs (TileShape.replaceAxisIndex shape ax out i)
      (trees pr (TileShape.axisDim shape ax) leaves).tree.eval (.literal pr .fp32 0)
        (binary contract pr .fp32 .add) leaves

/-- Erased real carrier casts inside fp32 registers are identities. A store
cast remains a node and therefore prevents contraction across kernel memory. -/
def cast (pr : Option ComputeDType) (src dst : FloatDType) (x : Expr) : Expr :=
  if pr=some .fp32 ∧ dst=.real ∧ (src=.fp32 ∨ src=.real) then x
  else .cast pr src dst x

def algebra (contract : Contraction) (trees : Trees) : Algebra Expr where
  literal := Expr.literal
  negInf := .word (FP.Value.infinity .fp32 Bool.true)
  binary := binary contract
  unary := fun pr op x => match op with
    | .rsqrt => .rsqrt pr x
    | .neg _ => .neg x
    | _ => .word (FP.Value.qNaN .fp32)
  cast := cast
  fromNat := fun pr n => .literal pr .fp32 (n : ℝ)
  fromInt := fun pr n => .literal pr .fp32 (n : ℝ)
  fp32Bits := fun bits => .word ⟨bits.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .word (FP.Value.qNaN .fp32)
  reduceSum := sum contract trees

def sums (contract : Contraction) (trees : Trees) : SumPrimitive Expr :=
  ⟨sum contract trees⟩

def hostScalar (_contract : Contraction) : Host.Scalar Expr :=
  ⟨.literal none .fp32 0,Expr.hostAdd,Expr.hostCast⟩

def evalCell (P : Primitives) : Cell Expr → Cell (FP.Value .fp32)
  | .mk .real v => .mk .real (v.eval P)
  | .mk .fp32 v => .mk .fp32 (v.eval P)
  | .mk .fp16 v => .mk .fp16 (v.eval P)
  | .mk .bf16 v => .mk .bf16 (v.eval P)
  | .mk .f8e4 v => .mk .f8e4 (v.eval P)
  | .mk .f8e5 v => .mk .f8e5 (v.eval P)
  | .mk .nat v => .mk .nat v
  | .mk .int v => .mk .int v
  | .mk .bool v => .mk .bool v
  | .mk .ptr v => .mk .ptr v
  | .mk .blockPtr v => .mk .blockPtr v

/-- Complete selected source-plan execution through both grids and Torch
axis-sum/cast return. The result contains only scalar-expression nodes. -/
theorem complete (contract : Contraction) (trees : Trees)
    (fp : Forward.Parameters) (bp : Backward.Parameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State Expr) :
    Host.forwardBackward (algebra contract trees) (sums contract trees) (hostScalar contract)
      fp bp rows programs pw pb dw db shape s=
      some (Host.finish (hostScalar contract) bp programs pw pb dw db shape
        (Grid.reference (Backward.reference (algebra contract trees) (sums contract trees) bp)
          (Grid.reference (Forward.reference (algebra contract trees) (sums contract trees) fp) s rows)
          programs)) :=
  Host.forwardBackward_complete _ _ _ _ _ _ _ _ _ _ _ _ _

/-- Concrete final output bits are obtained by evaluating every scalar node
in the actual returned memory, including host tree additions and casts. -/
theorem complete_bits (P : Primitives) (contract : Contraction) (trees : Trees)
    (fp : Forward.Parameters) (bp : Backward.Parameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State Expr) :
    (Host.forwardBackward (algebra contract trees) (sums contract trees) (hostScalar contract)
      fp bp rows programs pw pb dw db shape s).map
        (fun out => fun r o => evalCell P (out.state.mem r o))=
      some (fun r o => evalCell P ((Host.finish (hostScalar contract) bp programs pw pb dw db shape
        (Grid.reference (Backward.reference (algebra contract trees) (sums contract trees) bp)
          (Grid.reference (Forward.reference (algebra contract trees) (sums contract trees) fp) s rows)
          programs)).state.mem r o)) := by
  rw [complete]
  rfl

#axiomsClean fma_add_ideal
#axiomsClean fma_sub_ideal
#axiomsClean complete
#axiomsClean complete_bits
#specNonCircular Expr.eval avoiding [Forward.kernel,Backward.kernel,Forward.reference,Backward.reference]
end
end VeriTile.Bench.Repairs.LayerNormLiger.NativePlan
