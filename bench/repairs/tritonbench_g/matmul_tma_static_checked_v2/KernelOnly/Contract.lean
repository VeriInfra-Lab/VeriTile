import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Addresses

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open Host

/-- The complete frontend-static domain: original five explicit guards plus
the three complete block-tensor element limits imposed by Triton. -/
structure InputContract (h : Shape) (cfg : Config) extends StaticChecked.InputContract h cfg : Prop where
  aTile : h.rows*h.inner ≤ 1048576
  bTile : h.inner*h.columns ≤ 1048576
  cTile : h.rows*h.columns ≤ 1048576

def checkInputs (h : Shape) (cfg : Config) : Except String Unit :=
  match StaticChecked.checkInputs h cfg with
  | .error reason => .error reason
  | .ok () =>
    if h.rows*h.inner > 1048576 then .error "A block tile must contain at most 1048576 elements"
    else if h.inner*h.columns > 1048576 then .error "B block tile must contain at most 1048576 elements"
    else if h.rows*h.columns > 1048576 then .error "C block tile must contain at most 1048576 elements"
    else .ok ()

theorem checkInputs_iff (h : Shape) (cfg : Config) :
    checkInputs h cfg=.ok () ↔ InputContract h cfg := by
  have contract : InputContract h cfg ↔ StaticChecked.InputContract h cfg ∧
      h.rows*h.inner≤1048576 ∧ h.inner*h.columns≤1048576 ∧ h.rows*h.columns≤1048576 := by
    constructor
    · intro d;exact ⟨d.toInputContract,d.aTile,d.bTile,d.cTile⟩
    · rintro ⟨a,b,c,d⟩;exact ⟨a,b,c,d⟩
  rw [contract,←StaticChecked.checkInputs_iff]
  cases hc : StaticChecked.checkInputs h cfg with
  | error reason => simp [checkInputs,hc]
  | ok result => cases result; simp [checkInputs,hc,ite_eq_iff,and_assoc]

/-- Positive dimensions plus the source tile limits imply the signed32 lane
and scalar bounds; the public native endpoint needs no extra dimension gate. -/
theorem dimension_bounds (h : Shape) (cfg : Config) (d : InputContract h cfg) :
    h.rows≤1048576 ∧ h.columns≤1048576 ∧ h.inner≤1048576 := by
  have hm := Nat.pos_of_isPowerOfTwo d.rows
  have hn := Nat.pos_of_isPowerOfTwo d.columns
  have hk := Nat.pos_of_isPowerOfTwo d.inner
  have a := d.aTile
  have b := d.bTile
  constructor
  · nlinarith
  constructor <;> nlinarith

theorem dimensions (h : Shape) (cfg : Config) (d : InputContract h cfg) :
    StaticChecked.Addresses.Dimensions h := by
  obtain ⟨hm,hn,hk⟩ := dimension_bounds h cfg d
  unfold StaticChecked.Addresses.Dimensions
  norm_num
  omega

#axiomsClean checkInputs_iff
#axiomsClean dimensions
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
