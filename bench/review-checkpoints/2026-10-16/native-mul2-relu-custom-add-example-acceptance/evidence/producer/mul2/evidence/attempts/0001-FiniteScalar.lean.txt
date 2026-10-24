import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteCodec

namespace TritonMul2Finite
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

inductive Width where
  | w8 | w16 | w32 | w64
  deriving DecidableEq, Repr

def Width.bytes : Width → Nat
  | .w8 => 1 | .w16 => 2 | .w32 => 4 | .w64 => 8

inductive FloatFormat where
  | fp16 | bf16 | fp32 | fp64
  deriving DecidableEq, Repr

def FloatFormat.bytes : FloatFormat → Nat
  | .fp16 | .bf16 => 2 | .fp32 => 4 | .fp64 => 8

inductive Storage where
  | boolean | signed (w : Width) | unsigned (w : Width) | floating (f : FloatFormat)
  deriving DecidableEq, Repr

def Storage.bytes : Storage → Nat
  | .boolean => 1 | .signed w | .unsigned w => w.bytes | .floating f => f.bytes

abbrev Word (d : Storage) := BitVec (d.bytes*8)

def Storage.registerBytes : Storage → Nat
  | .boolean => 4 | d => d.bytes

abbrev Register (d : Storage) := BitVec (d.registerBytes*8)

/-- Only native floating multiplication is numerically deferred. Integer
multiplication, boolean raw-byte normalization and stores are concrete words. -/
structure Numerics where
  multiply : (f : FloatFormat) → BitVec (f.bytes*8) → BitVec (f.bytes*8) → BitVec (f.bytes*8)

def floatingTwo : (f : FloatFormat) → BitVec (f.bytes*8)
  | .fp16 => 0x4000#16 | .bf16 => 0x4000#16
  | .fp32 => 0x40000000#32 | .fp64 => 0x4000000000000000#64

def loaded : (d : Storage) → Word d → Register d
  | .boolean,x => if x=0 then 0 else 1
  | .signed _,x => x | .unsigned _,x => x | .floating _,x => x

def literalTwo : (d : Storage) → Register d
  | .boolean => 2 | .signed _ => 2 | .unsigned _ => 2 | .floating f => floatingTwo f

def multiply (b : Numerics) : (d : Storage) → Register d → Register d → Register d
  | .boolean,x,y => x*y
  | .signed _,x,y => x*y
  | .unsigned _,x,y => x*y
  | .floating f,x,y => b.multiply f x y

def stored : (d : Storage) → Register d → Word d
  | .boolean,x => x.truncate 8
  | .signed _,x => x | .unsigned _,x => x | .floating _,x => x

inductive Expr where
  | input | two | mul (left right : Expr)
  deriving DecidableEq

def Expr.eval (b : Numerics) (d : Storage) (x : Word d) : Expr → Register d
  | .input => loaded d x
  | .two => literalTwo d
  | .mul left right => multiply b d (left.eval b d x) (right.eval b d x)

/-- Exact unchanged scalar expression 2*x, with its actual load and store casts. -/
def source : Expr := .mul .two .input

def result (b : Numerics) : (d : Storage) → Word d → Word d
  | .boolean,x => if x=0 then 0 else 2
  | .signed _,x => 2*x
  | .unsigned _,x => 2*x
  | .floating f,x => b.multiply f (floatingTwo f) x

theorem source_scalar (b : Numerics) (d : Storage) (x : Word d) :
    stored d (source.eval b d x)=result b d x := by
  cases d with
  | boolean =>
      change (2#32 * (if x=0#8 then 0#32 else 1#32)).truncate 8 =
        (if x=0#8 then 0#8 else 2#8)
      by_cases h : x=0#8 <;> simp [h]
  | signed w => rfl
  | unsigned w => rfl
  | floating f => rfl

theorem bool_raw_store (b : Numerics) (x : BitVec 8) :
    result b .boolean x=(if x=0 then 0 else 2) := rfl

theorem storage_width_positive (d : Storage) : 0<d.bytes := by
  cases d with
  | boolean => decide
  | signed w | unsigned w => cases w <;> decide
  | floating f => cases f <;> decide

#specNonCircular result avoiding [source,Expr.eval,stored,multiply,loaded]
#axiomsClean source_scalar
#axiomsClean bool_raw_store
#axiomsClean storage_width_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end TritonMul2Finite
