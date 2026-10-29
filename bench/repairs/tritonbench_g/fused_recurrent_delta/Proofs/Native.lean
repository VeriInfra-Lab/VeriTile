import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
import VeriTile.Triton.Float.ScalarOps
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Codec
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Native
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
abbrev Word := FP.Value .fp32
@[ext] structure Instructions where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
def software : Instructions := ⟨FP.add .ieeeValue,FP.sub .ieeeValue,FP.mul .ieeeValue,FP.fma .ieeeValue⟩
/-- Four scalar leaves only. Tree routing, masks, loop results, memory and
host outputs are not numerical equality premises. -/
structure Obligations (actual : Instructions) : Prop where
  add : ∀ x y,actual.add x y=software.add x y
  sub : ∀ x y,actual.sub x y=software.sub x y
  mul : ∀ x y,actual.mul x y=software.mul x y
  fma : ∀ x y z,actual.fma x y z=software.fma x y z
inductive Expr where
  | bits (x : Word)
  | input (address : Addr) (x : Word)
  | stored (bytes : Codec.Cell)
  | zero | one
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | cut (a : Expr)
inductive Fusion where
  | separate | right | left
  deriving DecidableEq
/-- Value-erased graph shape. Cuts are opaque memory-rounding boundaries.
Routing cannot inspect tensor bits, NaN payloads, or floating results. -/
inductive Code where
  | literal | input (address : Addr) | zero | one | cut
  | add (a b : Code) | sub (a b : Code) | mul (a b : Code)
def Expr.code : Expr → Code
  | .bits _ => .literal
  | .input a _ => .input a
  | .stored _ => .cut
  | .zero => .zero
  | .one => .one
  | .cut _ => .cut
  | .add a b => .add a.code b.code
  | .sub a b => .sub a.code b.code
  | .mul a b => .mul a.code b.code
abbrev Policy := Code → Fusion
def Expr.eval (p : Instructions) (policy : Policy) : Expr → Word
  | .bits x => x
  | .input _ x => x
  | .stored bytes => Codec.decode bytes
  | .zero => FP.Value.zero .fp32
  | .one => FP.round .fp32 .ieeeValue 1
  | .mul a b => p.mul (a.eval p policy) (b.eval p policy)
  | .cut a => a.eval p policy
  | .add a b =>
    if policy (.add a.code b.code)=.right then
      match hb : b with
      | .mul u v => p.fma (u.eval p policy) (v.eval p policy) (a.eval p policy)
      | _ => p.add (a.eval p policy) (b.eval p policy)
    else if policy (.add a.code b.code)=.left then
      match ha : a with
      | .mul u v => p.fma (u.eval p policy) (v.eval p policy) (b.eval p policy)
      | _ => p.add (a.eval p policy) (b.eval p policy)
    else p.add (a.eval p policy) (b.eval p policy)
  | .sub a b =>
    if policy (.sub a.code b.code)=.right then
      match hb : b with
      | .mul u v => p.fma (u.eval p policy).neg (v.eval p policy) (a.eval p policy)
      | _ => p.sub (a.eval p policy) (b.eval p policy)
    else p.sub (a.eval p policy) (b.eval p policy)

termination_by e => sizeOf e
decreasing_by all_goals simp_all <;> omega
def graphOps : Ops Expr := ⟨.zero,.one,.add,.sub,.mul,.cut⟩
def graphMemory (m : Memory Word) : Memory Expr := fun a => .input a (m a)
def evaluate (p : Instructions) (policy : Policy) (m : Memory Expr) : Memory Word :=
  fun a => (m a).eval p policy
def forward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (policy : Policy) (m : Memory Word) :=
  evaluate p policy (forwardCall s f graphOps g (.bits scale) (graphMemory m))
def backward (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (policy : Policy) (m : Memory Word) :=
  evaluate p policy (backwardCall s f graphOps g (.bits scale) (graphMemory m))

theorem local_transport (p : Instructions) (h : Obligations p) (policy : Policy) (e : Expr) :
    e.eval p policy=e.eval software policy := by
  have heq : p=software := by
    apply Instructions.ext
    · exact funext fun x => funext fun y => h.add x y
    · exact funext fun x => funext fun y => h.sub x y
    · exact funext fun x => funext fun y => h.mul x y
    · exact funext fun x => funext fun y => funext fun z => h.fma x y z
  rw [heq]
theorem memory_transport (p : Instructions) (h : Obligations p) (policy : Policy) (m : Memory Expr) :
    evaluate p policy m=evaluate software policy m := by
  funext a
  exact local_transport p h policy (m a)
theorem forward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (h : Obligations p) (policy : Policy) (m : Memory Word) :
    forward s f g scale p policy m=forward s f g scale software policy m :=
  memory_transport p h policy _
theorem backward_transport (s : Shape) (f : Flags) (g : Trees) (scale : Word) (p : Instructions)
    (h : Obligations p) (policy : Policy) (m : Memory Word) :
    backward s f g scale p policy m=backward s f g scale software policy m :=
  memory_transport p h policy _

/-- A concrete K=1 residual lowering, including the negative sign at the
fixed-arity FMA leaf; an add-only reduction model cannot represent this. -/
theorem single_key_residual (p : Instructions) (x h k : Word) (policy : Policy)
    (route : policy (.sub .literal (.mul .literal .literal))=.right) :
    (Expr.sub (.bits x) (.mul (.bits h) (.bits k))).eval p policy=p.fma h.neg k x := by
  simp [Expr.eval,Expr.code,route]

#axiomsClean local_transport
#axiomsClean memory_transport
#axiomsClean forward_transport
#axiomsClean backward_transport
#axiomsClean single_key_residual
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Native
