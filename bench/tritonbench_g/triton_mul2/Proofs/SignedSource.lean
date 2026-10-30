import bench.tritonbench_g.triton_mul2.Proofs.FiniteSource

/-! Structural decoder for explicit signed index casts into the existing finite
index instruction vocabulary. It adds no arithmetic interpretation or bounds
lemma. Callers retain their original width/sign-extension and address proofs. -/
namespace TritonMul2Finite.SignedSource
open VeriTile.Triton

mutual
  def lowerNat {shape : TileShape} : Op .nat shape → Option IndexExpr
    | .constNat n => some (.literal n)
    | .programId ax => some (.program ax)
    | .ref _ _ name => some (.register name)
    | .arange width => some (.lane width)
    | .add _ _ a b => return .add (← lowerNat a) (← lowerNat b)
    | .mul _ _ a b => return .mul (← lowerNat a) (← lowerNat b)
    | .castIntToNat a => lowerInt a
    | _ => none

  def lowerInt {shape : TileShape} : Op .int shape → Option IndexExpr
    | .constInt n => if 0 ≤ n then some (.literal n.toNat) else none
    | .ref _ _ name => some (.register name)
    | .castNatToInt a => lowerNat a
    | .add _ _ a b => return .add (← lowerInt a) (← lowerInt b)
    | .mul _ _ a b => return .mul (← lowerInt a) (← lowerInt b)
    | _ => none
end

def lowerBool {shape : TileShape} : Op .bool shape → Option MaskExpr
  | .ref _ _ name => some (.register name)
  | .lt .nat _ a b => return .less (← lowerNat a) (← lowerNat b)
  | .lt .int _ a b => return .less (← lowerInt a) (← lowerInt b)
  | _ => none

def lowerMask {shape : TileShape} {dtype : TileDType} : MaskOpt dtype shape → Option MaskExpr
  | .mask e => lowerBool e
  | _ => none

end TritonMul2Finite.SignedSource
