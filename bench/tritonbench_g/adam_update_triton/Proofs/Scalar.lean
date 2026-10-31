import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.adam_update_triton.Kernels
namespace VeriTile.Bench.Repairs.AdamUpdateTriton
theorem source_body_length (total chunk block : Nat) :
    (source total chunk block).surfaceBody.length = 19 := by rfl

end VeriTile.Bench.Repairs.AdamUpdateTriton

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Scalar
noncomputable section

/-- Payload classification follows the closed v003 dispatch. Integer includes
bool 0/1 and int subclasses; binary64 includes float subclasses/np.float64.
An exact tl.constexpr wrapper contributes its stored payload once. Tensor and
arbitrary coercible objects, including unsupported wrapper payloads, fail. -/
inductive Payload where
  | integer (value : Int)
  | binary64 (bits : BitVec 64)
  | unsupported

inductive Argument where
  | plain (value : Payload)
  | constexpr (value : Payload)

def payload : Argument → Payload
  | .plain value | .constexpr value => value

inductive Name where
  | lr | wd | beta1 | beta2
  deriving DecidableEq

structure Failure where
  argument : Name
  reason : String

/-- The two actual local conversion boundaries stay separate. In particular,
integer→Python binary64→FP32 may double-round. This interface asserts neither
direct-int rounding nor finite/range restrictions nor software NaN identities. -/
structure Conversions (α : Type) where
  integerToPythonFloat : Int → Except String (BitVec 64)
  pythonFloatToFP32 : BitVec 64 → Except String α

def normalizeOne (convert : Conversions α) (name : Name) (input : Argument) :
    Except Failure (BitVec 64) :=
  match payload input with
  | .unsupported => .error ⟨name,"unsupported numeric scalar"⟩
  | .binary64 bits => .ok bits
  | .integer value => (convert.integerToPythonFloat value).mapError (Failure.mk name)

/-- Exactly the source order, before the CUDA context and direct caller writes. -/
def normalize (convert : Conversions α) (args : Scalars Argument) :
    Except Failure (Scalars (BitVec 64)) := do
  let lr ← normalizeOne convert .lr args.lr
  let wd ← normalizeOne convert .wd args.wd
  let beta1 ← normalizeOne convert .beta1 args.beta1
  let beta2 ← normalizeOne convert .beta2 args.beta2
  pure ⟨lr,wd,beta1,beta2⟩

/-- Actual launch setup consumes Python floats, hence all four scalar ABI
types are FP32. Setup/binder failure remains before the corresponding in-place launch. -/
def bind (convert : Conversions α) (args : Scalars (BitVec 64)) : Except Failure (Scalars α) := do
  let lr ← (convert.pythonFloatToFP32 args.lr).mapError (Failure.mk .lr)
  let wd ← (convert.pythonFloatToFP32 args.wd).mapError (Failure.mk .wd)
  let beta1 ← (convert.pythonFloatToFP32 args.beta1).mapError (Failure.mk .beta1)
  let beta2 ← (convert.pythonFloatToFP32 args.beta2).mapError (Failure.mk .beta2)
  pure ⟨lr,wd,beta1,beta2⟩

theorem wrapper_transparent (convert : Conversions α) (name : Name) (value : Payload) :
    normalizeOne convert name (.constexpr value)=normalizeOne convert name (.plain value) := rfl

theorem unsupported_failure (convert : Conversions α) (name : Name) :
    normalizeOne convert name (.plain .unsupported)=.error ⟨name,"unsupported numeric scalar"⟩ := rfl

theorem binary64_retained (convert : Conversions α) (name : Name) (bits : BitVec 64) :
    normalizeOne convert name (.plain (.binary64 bits))=.ok bits := rfl

#axiomsClean wrapper_transparent
#axiomsClean binary64_retained
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Scalar
