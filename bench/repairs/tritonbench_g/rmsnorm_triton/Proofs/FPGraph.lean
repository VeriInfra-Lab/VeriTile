import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelPublic
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPConstants

namespace VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

/-- Floating syntax retained by actual Structural execution, before native
instruction selection. Reduction inputs and integer power remain explicit. -/
inductive Term where
  | input (word : VeriTile.Torch.FP32.Word)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Term)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Term)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Term)
  | powi (precision : Option ComputeDType) (value : Term) (exponent : Int)
  | sum (precision : Option ComputeDType) (width : Nat) (values : Fin width → Term)
  | unused
instance : Inhabited Term := ⟨.unused⟩

def graphSum (p : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Term .real shape) : Values Term .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => .sum p (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.insertAxisIndex shape ax out j))
  · exact fun out => .sum p (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.replaceAxisIndex shape ax out j))

def algebra : Algebra Term where
  literal := .literal
  negInf := .unused
  binary := .binary
  unary := .unary
  cast := .cast
  libdevicePowi := fun p => some (.powi p)
  fromNat := fun p n => .literal p .real n
  fromInt := fun p n => .literal p .real n
  fp32Bits := fun w => .input ⟨w.bits⟩
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => .unused
  reduceSum := fun p {_} ax keep xs => graphSum p ax keep xs

def lit (x : ℝ) : Term := .literal (some .fp32) .real x
def cast32 (x : Term) : Term := .cast (some .fp32) .real .real x
def add (x y : Term) : Term := .binary (some .fp32) .real .add x y
def mul (x y : Term) : Term := .binary (some .fp32) .real .mul x y
def divide (x y : Term) : Term := .binary (some .fp32) .real .div x y
def rsqrt (x : Term) : Term := .unary (some .fp32) .rsqrt x
def square (x : Term) : Term := .powi (some .fp32) x 2

def entryGraph (before : Memory VeriTile.Torch.FP32.Word) : Memory Term :=
  fun allocation offset => .input (before allocation offset)

/-- The supported libdevice integer-power lowering keeps both FTZ multiplies,
including the leading multiplication by one. -/
inductive PowTwoPolicy where
  | oneTimesSquareFTZ

/-- Native choices depend on operation structure, never on input payloads.
The reduction layout, initial add order, output operand order, integer-power
lowering, and width-one division specialization are all explicit. -/
structure Profile where
  vector : Bool
  multi : Bool
  initialRightZero : Bool
  swapRsqrtProduct : Bool
  powTwo : PowTwoPolicy := .oneTimesSquareFTZ
  elideWidthOne : Bool := Bool.true

structure Lowered where
  value : VeriTile.Torch.FP32.Word
  constant : Option ℝ := none
  power : Bool := Bool.false
  reciprocalSqrt : Bool := Bool.false

def lower (Q : Numerics VeriTile.Torch.FP32.Word) (profile : Profile) : Term → Lowered
  | .input word => ⟨word,none,Bool.false,Bool.false⟩
  | .literal _ .real value => ⟨Q.literal value,some value,Bool.false,Bool.false⟩
  | .cast _ .real .real value => lower Q profile value
  | .powi (some .fp32) value 2 =>
      let v := (lower Q profile value).value
      match profile.powTwo with
      | .oneTimesSquareFTZ => ⟨Q.mulFTZ (Q.literal 1) (Q.mulFTZ v v),none,Bool.true,Bool.false⟩
  | .binary (some .fp32) .real .add left right =>
      let a := lower Q profile left
      let b := lower Q profile right
      ⟨if profile.initialRightZero && decide (a.constant=some 0) && b.power
        then Q.add b.value a.value else Q.add a.value b.value,none,Bool.false,Bool.false⟩
  | .binary (some .fp32) .real .mul left right =>
      let a := lower Q profile left
      let b := lower Q profile right
      ⟨if profile.swapRsqrtProduct && b.reciprocalSqrt then Q.mul b.value a.value
        else Q.mul a.value b.value,none,Bool.false,Bool.false⟩
  | .binary (some .fp32) .real .div left right =>
      let a := lower Q profile left
      let b := lower Q profile right
      if profile.elideWidthOne && decide (b.constant=some 1) then a else ⟨Q.div a.value b.value,none,Bool.false,Bool.false⟩
  | .unary (some .fp32) .rsqrt value =>
      ⟨Q.rsqrt (lower Q profile value).value,none,Bool.false,Bool.true⟩
  | .sum (some .fp32) width values =>
      if h : width=1024 then
        ⟨(nativeTree profile.vector profile.multi).eval Q.add
          (fun i => (lower Q profile (values ⟨i.val,by simpa [h] using i.isLt⟩)).value),none,Bool.false,Bool.false⟩
      else ⟨Q.literal 0,none,Bool.false,Bool.false⟩
  | _ => ⟨Q.literal 0,none,Bool.false,Bool.false⟩

def observe (Q : Numerics VeriTile.Torch.FP32.Word) (profile : Profile) (term : Term) :=
  (lower Q profile term).value

def projectMemory (Q : Numerics VeriTile.Torch.FP32.Word) (profile : Profile) (memory : Memory Term) :
    Memory VeriTile.Torch.FP32.Word := fun a i => observe Q profile (memory a i)

@[simp] theorem observe_input (Q : Numerics VeriTile.Torch.FP32.Word) (profile : Profile)
    (word : VeriTile.Torch.FP32.Word) : observe Q profile (.input word)=word := rfl

end
end VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
