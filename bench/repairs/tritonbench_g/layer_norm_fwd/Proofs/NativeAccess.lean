import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.PhysicalDomain

/-! Active scalar accesses must not depend on a vector lane existing. In
particular N=0 still executes scalar seed/rowscale loads and mean/rstd stores
on each launched row. These lemmas close that coverage hole; source/native
control-flow composition remains a separate obligation. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeAccess
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain
noncomputable section

def laneActive (N col : Nat) : Role → Prop
  | .mean | .rstd | .rowscale | .rowscale1 | .seeds | .seeds1 => True
  | _ => col < N

theorem all_roles_bounded (f : Flags) (r : Regions) (stride : Strides) (M N row col : Nat)
    (bounds : RegionBounds) (h : FlatDomain.RowBounds f r stride M N row bounds)
    (role : Role) (ha : active f role) (hl : laneActive N col role) :
    offset stride M N row col role < bounds (region r role) := by
  cases role with
  | x => exact h.x col hl
  | y => exact h.y col hl
  | residual => exact h.residual ha col hl
  | residualOut => exact h.residualOut ha col hl
  | x1 => exact h.x1 ha col hl
  | y1 => exact h.y1 ha col hl
  | weight => exact h.weight col hl
  | bias => exact h.bias ha col hl
  | weight1 => exact h.weight1 ha col hl
  | bias1 => exact h.bias1 ha.1 ha.2 col hl
  | mean => exact h.mean ha
  | rstd => exact h.rstd
  | rowscale => exact h.rowscale ha
  | rowscale1 => exact h.rowscale1 ha.1 ha.2
  | seeds => exact h.seeds ha
  | seeds1 => exact h.seeds1 ha.1 ha.2
  | mask => exact h.mask ha.1 ha.2 col hl
  | mask1 => exact h.mask1 ha.1 ha.2.1 ha.2.2 col hl

theorem offset_exact (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation) (h : PhysicalDomain.Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (role : Role)
    (ha : active f role) (hl : laneActive N col role) :
    machineOffset stride M N row col role=offset stride M N row col role := by
  have hb := all_roles_bounded f r stride M N row col _
    (h.accesses row hr) role ha hl
  have he := (a (region r role)).signed_extent
  have hrow : row<2^63 := lt_trans hr h.metadata.rows
  cases role with
  | x => exact matrix_machine_exact hrow h.metadata.x hb he
  | y => exact matrix_machine_exact hrow h.metadata.y hb he
  | residual => exact matrix_machine_exact hrow h.metadata.residual hb he
  | residualOut => exact matrix_machine_exact hrow h.metadata.residualOut hb he
  | x1 => exact matrix_machine_exact hrow h.metadata.x1 hb he
  | y1 => exact matrix_machine_exact hrow h.metadata.y1 hb he
  | mask => exact matrix_machine_exact hrow h.metadata.cols hb he
  | mask1 => exact mask_machine_exact (by change col<N at hl; omega) h.metadata.cols hb he
  | weight | bias | weight1 | bias1 =>
    exact wrap64_exact (lt_trans hl h.metadata.cols)
  | mean | rstd | rowscale | seeds => exact wrap64_exact hrow
  | rowscale1 | seeds1 =>
    have hh : M+row<2^63 := lt_of_lt_of_le hb he
    simp only [machineOffset,offset,add64,wrap64_exact h.metadata.rows,
      wrap64_exact hrow,wrap64_exact hh]

def AllMachineAddresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation) : Prop :=
  ∀ row col, row<M → ∀ role, active f role → laneActive N col role →
    machineByteAddress a r stride M N row col role =
      (layout a).address (region r role,offset stride M N row col role)

theorem all_machine_addresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation) (h : PhysicalDomain.Domain f r stride M N a) :
    AllMachineAddresses f r stride M N a := by
  intro row col hr role ha hl
  have he := offset_exact f r stride M N a h row col hr role ha hl
  have hb := all_roles_bounded f r stride M N row col _
    (h.accesses row hr) role ha hl
  have hbyte := (byte_address (a (region r role)) hb).2.2.1
  change (a (region r role)).base + offset stride M N row col role *
    (a (region r role)).width < 2^64 at hbyte
  have hbase : (a (region r role)).base<2^64 := by omega
  have hmul : offset stride M N row col role * (a (region r role)).width<2^64 := by omega
  simp only [machineByteAddress,he,←h.widths role ha,add64,mul64,
    wrap64,Nat.mod_eq_of_lt hbase,Nat.mod_eq_of_lt hmul,Nat.mod_eq_of_lt hbyte,
    layout,byteAddress]

theorem zero_columns_rstd (f : Flags) (r : Regions) (stride : Strides) (M : Nat)
    (a : RegionName → Allocation) (h : PhysicalDomain.Domain f r stride M 0 a)
    (row : Nat) (hr : row<M) :
    machineByteAddress a r stride M 0 row 0 .rstd =
      (layout a).address (r.rstd,row) :=
  all_machine_addresses f r stride M 0 a h row 0 hr .rstd trivial trivial

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeAccess
