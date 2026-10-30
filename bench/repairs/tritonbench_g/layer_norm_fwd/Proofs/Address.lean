import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.SourceBinding
import bench.tritonbench_g.layer_norm_fwd.Proofs.FlatDomain
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormFwd.Address
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

/-- Two's-complement signed decoding of an i64 arithmetic result. -/
def signed64 (n : Nat) : Int :=
  if n % 2^64 < 2^63 then (n % 2^64 : Nat) else (n % 2^64 : Nat) - (2^64 : Int)

theorem signed64_exact {n : Nat} (h : n < 2^63) : signed64 n = (n : Int) := by
  have hn : n < 2^64 := lt_trans h (by norm_num)
  simp only [signed64, Nat.mod_eq_of_lt hn, if_pos h]

/-- Multiplication and the following column addition both stay signed-i64.
The bound comes from an allocated tensor's finite span, not an i32 guard. -/
theorem matrix_intermediates {row stride col extent : Nat}
    (h : row*stride+col < extent) (he : extent ≤ 2^63) :
    signed64 (row*stride) = (row*stride : Nat) ∧
    signed64 (row*stride+col) = (row*stride+col : Nat) := by
  have hm : row*stride < 2^63 := by omega
  exact ⟨signed64_exact hm, signed64_exact (by omega)⟩

/-- Covers both streams: `M + row`, its product by N and final masked col. -/
theorem mask_intermediates {M row N col extent : Nat}
    (hN : 0 < N) (h : (M+row)*N+col < extent) (he : extent ≤ 2^63) :
    signed64 (M+row) = (M+row : Nat) ∧
    signed64 ((M+row)*N) = ((M+row)*N : Nat) ∧
    signed64 ((M+row)*N+col) = ((M+row)*N+col : Nat) := by
  have hm : M+row ≤ (M+row)*N := Nat.le_mul_of_pos_right _ hN
  exact ⟨signed64_exact (by omega), matrix_intermediates h he⟩

/-- Every allocated active address, including element-to-byte conversion,
fits its allocation and unsigned native pointer width. Byte bases include the
view storage offset. This covers FP32, int64 seeds and bool masks separately. -/
structure Allocation where
  base : Nat
  extent : Nat
  width : Nat
  width_pos : 0 < width
  signed_extent : extent ≤ 2^63
  bytes_finite : base + extent*width ≤ 2^64

def byteAddress (a : Allocation) (offset : Nat) : Nat := a.base + offset*a.width

theorem byte_address (a : Allocation) {offset : Nat} (h : offset < a.extent) :
    a.base ≤ byteAddress a offset ∧
    byteAddress a offset < a.base+a.extent*a.width ∧
    byteAddress a offset < 2^64 ∧
    signed64 offset = (offset : Int) := by
  have hm := Nat.mul_lt_mul_of_pos_right h a.width_pos
  refine ⟨by simp [byteAddress], by simpa [byteAddress] using Nat.add_lt_add_left hm a.base,
    ?_, signed64_exact (lt_of_lt_of_le h a.signed_extent)⟩
  unfold byteAddress
  have hf := a.bytes_finite
  omega

/-- Word-level operations for the actual widened expression. -/
def wrap64 (n : Nat) : Nat := n % 2^64
def add64 (a b : Nat) : Nat := wrap64 (a+b)
def mul64 (a b : Nat) : Nat := wrap64 (a*b)
def matrixOffset64 (row stride col : Nat) : Nat :=
  add64 (mul64 (wrap64 row) (wrap64 stride)) (wrap64 col)
def maskOffset64 (M row N col : Nat) : Nat :=
  add64 (mul64 (add64 (wrap64 M) (wrap64 row)) (wrap64 N)) (wrap64 col)

theorem wrap64_exact {n : Nat} (h : n < 2^63) : wrap64 n = n :=
  Nat.mod_eq_of_lt (lt_trans h (by norm_num))

/-- Nested word operations equal the natural offset, not merely its final
signed decoding. Metadata bounds are the tensor-library signed-i64 metadata
contract; access bounds follow from the actual allocated span. -/
theorem matrix_machine_exact {row stride col extent : Nat}
    (hr : row < 2^63) (hs : stride < 2^63)
    (ha : row*stride+col < extent) (he : extent ≤ 2^63) :
    matrixOffset64 row stride col = row*stride+col := by
  have hc : col < 2^63 := by omega
  have hm : row*stride < 2^63 := by omega
  have ht : row*stride+col < 2^63 := by omega
  simp only [matrixOffset64, wrap64_exact hr, wrap64_exact hs, wrap64_exact hc,
    mul64, add64, wrap64_exact hm, wrap64_exact ht]

theorem mask_machine_exact {M row N col extent : Nat}
    (hN : 0 < N) (hNb : N < 2^63)
    (ha : (M+row)*N+col < extent) (he : extent ≤ 2^63) :
    maskOffset64 M row N col = (M+row)*N+col := by
  have hmr : M+row ≤ (M+row)*N := Nat.le_mul_of_pos_right _ hN
  have hM : M < 2^63 := by omega
  have hr : row < 2^63 := by omega
  have hc : col < 2^63 := by omega
  have hs : M+row < 2^63 := by omega
  have hm : (M+row)*N < 2^63 := by omega
  have ht : (M+row)*N+col < 2^63 := by omega
  simp only [maskOffset64, wrap64_exact hM, wrap64_exact hr, wrap64_exact hNb,
    wrap64_exact hc, add64, wrap64_exact hs, mul64, wrap64_exact hm, wrap64_exact ht]

/-- Actual access roles, including separate second RNG and mask streams. -/
inductive Role where
  | x | y | residual | residualOut | x1 | y1 | weight | bias | weight1 | bias1
  | mean | rstd | rowscale | rowscale1 | seeds | seeds1 | mask | mask1
  deriving DecidableEq

def region (r : Regions) : Role → RegionName
  | .x => r.x | .y => r.y | .residual => r.residual | .residualOut => r.residualOut
  | .x1 => r.x1 | .y1 => r.y1 | .weight => r.weight | .bias => r.bias
  | .weight1 => r.weight1 | .bias1 => r.bias1 | .mean => r.mean | .rstd => r.rstd
  | .rowscale | .rowscale1 => r.rowscale
  | .seeds | .seeds1 => r.seeds | .mask | .mask1 => r.dropoutMask

def offset (stride : Strides) (M N row col : Nat) : Role → Nat
  | .x => row*stride.x+col | .y => row*stride.y+col
  | .residual => row*stride.residual+col | .residualOut => row*stride.residualOut+col
  | .x1 => row*stride.x1+col | .y1 => row*stride.y1+col
  | .weight | .bias | .weight1 | .bias1 => col
  | .mean | .rstd | .rowscale | .seeds => row
  | .rowscale1 | .seeds1 => M+row
  | .mask => row*N+col | .mask1 => (M+row)*N+col

def active (f : Flags) : Role → Prop
  | .x | .y | .weight | .rstd => True
  | .residual => f.residual = Bool.true | .residualOut => f.residualOut = Bool.true
  | .x1 => f.x1 = Bool.true | .y1 | .weight1 => f.w1 = Bool.true
  | .bias => f.bias = Bool.true | .bias1 => f.w1 = Bool.true ∧ f.b1 = Bool.true
  | .mean => f.rms = Bool.false | .rowscale => f.rowscale = Bool.true
  | .rowscale1 => f.x1 = Bool.true ∧ f.rowscale = Bool.true
  | .seeds => f.dropout = Bool.true | .seeds1 => f.x1 = Bool.true ∧ f.dropout = Bool.true
  | .mask => f.dropout = Bool.true ∧ f.dropoutMask = Bool.true
  | .mask1 => f.x1 = Bool.true ∧ f.dropout = Bool.true ∧ f.dropoutMask = Bool.true

theorem all_roles_bounded (f : Flags) (r : Regions) (stride : Strides) (M N row col : Nat)
    (bounds : RegionBounds) (h : FlatDomain.RowBounds f r stride M N row bounds)
    (hc : col < N) (role : Role) (ha : active f role) :
    offset stride M N row col role < bounds (region r role) := by
  cases role with
  | x => exact h.x col hc
  | y => exact h.y col hc
  | residual => exact h.residual ha col hc
  | residualOut => exact h.residualOut ha col hc
  | x1 => exact h.x1 ha col hc
  | y1 => exact h.y1 ha col hc
  | weight => exact h.weight col hc
  | bias => exact h.bias ha col hc
  | weight1 => exact h.weight1 ha col hc
  | bias1 => exact h.bias1 ha.1 ha.2 col hc
  | mean => exact h.mean ha
  | rstd => exact h.rstd
  | rowscale => exact h.rowscale ha
  | rowscale1 => exact h.rowscale1 ha.1 ha.2
  | seeds => exact h.seeds ha
  | seeds1 => exact h.seeds1 ha.1 ha.2
  | mask => exact h.mask ha.1 ha.2 col hc
  | mask1 => exact h.mask1 ha.1 ha.2.1 ha.2.2 col hc

/-- All actual grid read/write roles use representable finite addresses.
The allocation family is ordinary byte storage, while the existing execution
proof separately supplies whole-memory frames in finite tagged cells. -/
theorem all_grid_addresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation)
    (h : FlatDomain.Extents f r stride M N (fun q => (a q).extent))
    (row col : Nat) (hr : row < M) (hc : col < N)
    (role : Role) (ha : active f role) :
    signed64 (offset stride M N row col role) = (offset stride M N row col role : Int) ∧
    byteAddress (a (region r role)) (offset stride M N row col role) < 2^64 := by
  have hb := all_roles_bounded f r stride M N row col _ (h.rowBounds f r stride M N row _ hr) hc role ha
  have ht := byte_address (a (region r role)) hb
  exact ⟨ht.2.2.2, ht.2.2.1⟩

theorem repaired_strided_witness : signed64 (2*1073741824) = 2147483648 := by
  norm_num [signed64]
theorem repaired_contiguous_witness : signed64 (262144*8192) = 2147483648 := by
  norm_num [signed64]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Address
