import tests.lean.IntegerWhereReference
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.IntegerWhereLiterals
namespace VeriTile.Tests.IntegerWhereLiterals
open VeriTile.Triton IntegerWhereReference

def originalInteger (L Out X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    n = tl.load(L)
    bound = tl.where(n > 0, 1, 0)
    for i in range(0, bound, 1):
        tl.store(Out, i)
    return
  }

theorem integer_program : originalInteger=integerReference := rfl

def originalFloating (L Out X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = tl.load(X)
    y = tl.where(x > 0.0, 1.0, 0.0)
    tl.store(Y, y)
    return
  }

theorem floating_program : originalFloating=floatingReference := rfl
/-- Every selected weak literal already fits source i32 storage. The
integer carrier needs no extra narrowing instruction. -/
theorem selected_i32 (choose : Bool) (a b : Nat) (ha : a<2^31) (hb : b<2^31) :
    let n : Int := if choose then a else b
    (n+2147483648)%4294967296-2147483648=n := by
  dsimp
  split <;> rw [Int.emod_eq_of_lt (by omega) (by omega)] <;> omega

#axiomsClean integer_program
#axiomsClean floating_program
#axiomsClean selected_i32
end VeriTile.Tests.IntegerWhereLiterals
