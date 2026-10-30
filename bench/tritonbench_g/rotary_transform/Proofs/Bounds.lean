import bench.tritonbench_g.rotary_transform.Proofs.Host
namespace VeriTile.Bench.Repairs.RotaryTransform
/-- Checked source-index ABI. These are bounds on initial metadata, shape,
and padded expressions; no result-valued premise is present. -/
structure IndexABI (c : Config) : Prop where
  data : Host.size c.shape+8*c.shape.heads*c.shape.width+c.shape.blockK<2^31
  maximum : c.shape.maximum+8<2^31
  batches : c.shape.batches+1<2^31
  coefficients : c.shape.coefficientRows*c.shape.half<2^31
  offsets : ∀ b,b<c.shape.batches → c.shift b+c.shape.maximum+8<2^31
  paddedCoefficients : ∀ b,b<c.shape.batches → (c.shift b+c.shape.maximum+8)*c.shape.half+c.shape.blockK<2^31

theorem cell_bound (c : Config) (hc : c.Valid) (w : Work) (hw : c.Active w) (o : Bool) :
    c.cell w o<Host.size c.shape :=
  Host.address_bound c.shape _ _ _ (row_bound c hc _ _ hw.1 hw.2.1) hw.2.2.1
    (lt_of_lt_of_le (dim_bound c.shape w.pair hw.2.2.2 o) hc.rotary)
theorem coeff_bound (c : Config) (w : Work) (hw : c.Active w) (cached : c.hasCoeff w) :
    c.coeff w<c.shape.coefficientRows*c.shape.half := by
  unfold Config.coeff
  calc
    _ < (w.row+c.shift w.batch)*c.shape.half+c.shape.half := Nat.add_lt_add_left hw.2.2.2 _
    _ = (w.row+c.shift w.batch+1)*c.shape.half := by ring
    _ ≤ _ := Nat.mul_le_mul_right _ (by exact cached)

theorem cell_signed32 (c : Config) (hc : c.Valid) (abi : IndexABI c) (w : Work) (hw : c.Active w) (o : Bool) :
    c.cell w o<2^31 := by have := cell_bound c hc w hw o;have := abi.data;omega
theorem coeff_signed32 (c : Config) (abi : IndexABI c) (w : Work) (hw : c.Active w) (cached : c.hasCoeff w) :
    c.coeff w<2^31 := lt_trans (coeff_bound c w hw cached) abi.coefficients

theorem byte_signed64 (c : Config) (hc : c.Valid) (abi : IndexABI c) (w : Work) (hw : c.Active w)
    (o : Bool) (i : Fin 4) : 4*c.cell w o+i.val<2^63 := by
  have := cell_signed32 c hc abi w hw o
  omega
/-- Packed metadata reads use batch and batch+1, both inside CU[B+1]. -/
theorem cumulative_addresses (c : Config) (w : Work) (hw : c.Active w) :
    w.batch<c.shape.batches+1 ∧ w.batch+1<c.shape.batches+1 := by have := hw.1;omega
/-- The padded row offsets are bounded even when their payload load is masked. -/
theorem padded_row (c : Config) (pid : Program) (hp : pid∈programs c) (lane : Nat)
    (hl : lane<c.shape.blockM) : pid.1*c.shape.blockM+lane<c.shape.maximum+8 := by
  rcases pid with ⟨pm,pb,ph⟩
  simp only [programs,List.mem_product,List.mem_range] at hp
  dsimp only at hl ⊢
  have hm := hp.1
  unfold Shape.rowBlocks at hm
  rcases blockM_cases c.shape with bm|bm <;> rw [bm] at hm hl ⊢ <;> omega

theorem padded_coeff_signed32 (c : Config) (abi : IndexABI c) (pid : Program) (hp : pid∈programs c)
    (lane col : Nat) (hl : lane<c.shape.blockM) (hk : col<c.shape.blockK) :
    (pid.1*c.shape.blockM+lane+c.shift pid.2.1)*c.shape.half+col<2^31 := by
  have hr := padded_row c pid hp lane hl
  have hb : pid.2.1<c.shape.batches := by
    exact List.mem_range.mp (List.mem_product.mp (List.mem_product.mp hp).2).1
  have hi := abi.paddedCoefficients pid.2.1 hb
  have mul := Nat.mul_le_mul_right c.shape.half (show pid.1*c.shape.blockM+lane+c.shift pid.2.1≤c.shift pid.2.1+c.shape.maximum+8 by omega)
  omega
#axiomsClean cell_signed32
#axiomsClean coeff_signed32
#axiomsClean byte_signed64
#axiomsClean padded_coeff_signed32
end VeriTile.Bench.Repairs.RotaryTransform

namespace VeriTile.Bench.Repairs.RotaryTransform
/-- Every formed data pointer in a live program, including padded lanes,
fits the admitted signed32 domain. The extra block width includes the split
partner and the adjacent partner, before the payload masks are applied. -/
theorem padded_input_signed32 (c : Config) (hc : c.Valid) (abi : IndexABI c)
    (batch row head col : Nat) (hb : batch<c.shape.batches)
    (hr : row<c.length batch+8) (hh : head<c.shape.heads) (hk : col<c.shape.blockK) :
    c.shape.address (c.start batch+row) head col<2^31 := by
  have next := start_next c hc batch hb
  have bound := start_mono c hc (batch+1) c.shape.batches (by omega) (by omega)
  rw [start_total c hc] at bound
  have token : c.start batch+row<c.shape.total+8 := by omega
  have a : (c.start batch+row)*c.shape.heads+head<(c.shape.total+8)*c.shape.heads := by
    calc
      _ < (c.start batch+row)*c.shape.heads+c.shape.heads := Nat.add_lt_add_left hh _
      _ = (c.start batch+row+1)*c.shape.heads := by ring
      _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
  have b := Nat.mul_le_mul_right c.shape.width (Nat.le_of_lt a)
  have eq : ((c.shape.total+8)*c.shape.heads)*c.shape.width=
      Host.size c.shape+8*c.shape.heads*c.shape.width := by unfold Host.size;ring
  rw [eq] at b
  have lim := abi.data
  unfold Shape.address
  omega
#axiomsClean padded_input_signed32
end VeriTile.Bench.Repairs.RotaryTransform
