import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeABI

/-! Actual requested output widths. The integer/address proof uses each real
pointer descriptor's byte width; it never substitutes the FP32 width for an
explicit out_dtype/residual_dtype allocation. Input FP32 admission is unchanged. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 2000000

structure Domain (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (allocations : RegionName → Allocation) : Prop where
  metadata : Metadata stride M N
  accesses : ∀ row, row<M → FlatDomain.RowBounds f r stride M N row (fun region => (allocations region).extent)
  fresh : AliasMemory.FreshWrites (layout allocations) (AliasMemory.Writes f r stride M N)

theorem offset_exact (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (role : Role)
    (ha : active f role) (hl : NativeAccess.laneActive N col role) :
    machineOffset stride M N row col role=offset stride M N row col role := by
  have hb := NativeAccess.all_roles_bounded f r stride M N row col _ (h.accesses row hr) role ha hl
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
  | weight | bias | weight1 | bias1 => exact wrap64_exact (lt_trans hl h.metadata.cols)
  | mean | rstd | rowscale | seeds => exact wrap64_exact hrow
  | rowscale1 | seeds1 =>
    have hh : M+row<2^63 := lt_of_lt_of_le hb he
    simp only [machineOffset,offset,add64,wrap64_exact h.metadata.rows,wrap64_exact hrow,wrap64_exact hh]

def byteAddress (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (role : Role) : Nat :=
  add64 (wrap64 (a (region r role)).base)
    (mul64 (machineOffset stride M N row col role) (a (region r role)).width)

def nativeAddress (a : RegionName → Allocation) (r : Regions) (stride : Strides)
    (M N row col : Nat) (role : Role) : Nat :=
  byteAddress a r (strideWords stride) (metadataWord M) (metadataWord N)
    (signExtend32 row) (zeroExtend32 col) role

theorem byte_address_exact (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Allocation) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (role : Role) (ha : active f role) (hl : NativeAccess.laneActive N col role) :
    byteAddress a r stride M N row col role=(layout a).address (region r role,offset stride M N row col role) := by
  have he := offset_exact f r stride M N a h row col hr role ha hl
  have hb := NativeAccess.all_roles_bounded f r stride M N row col _ (h.accesses row hr) role ha hl
  have hbyte := (Address.byte_address (a (region r role)) hb).2.2.1
  change (a (region r role)).base+offset stride M N row col role*(a (region r role)).width<2^64 at hbyte
  have hbase : (a (region r role)).base<2^64 := by omega
  have hmul : offset stride M N row col role*(a (region r role)).width<2^64 := by omega
  simp only [byteAddress,he,add64,mul64,wrap64,Nat.mod_eq_of_lt hbase,Nat.mod_eq_of_lt hmul,
    Nat.mod_eq_of_lt hbyte,layout,Address.byteAddress]

theorem native_address_exact (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (hadmit : Accepted M N block) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (hg : machineGuard f N col role=Bool.true) :
    nativeAddress a r stride M N row col role=(layout a).address (region r role,offset stride M N row col role) := by
  obtain ⟨ha,hl⟩ := (machine_guard_iff hadmit hc f role).mp hg
  simp only [nativeAddress,stride_words_exact stride M N h.metadata,metadata_word_exact h.metadata.rows,
    metadata_word_exact h.metadata.cols,signExtend32_exact (accepted_row hadmit hr),
    zeroExtend32_exact (accepted_col hadmit hc)]
  exact byte_address_exact f r stride M N a h row col hr role ha hl

/-- Every byte of the actual requested element width is bounded, including
non-FP32 output widths. Scalar accesses still exist when N=0. -/
theorem native_access_span (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (hadmit : Accepted M N block) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<block) (role : Role)
    (hg : machineGuard f N col role=Bool.true) (byte : Nat) (hb : byte<(a (region r role)).width) :
    (a (region r role)).base≤nativeAddress a r stride M N row col role+byte ∧
    nativeAddress a r stride M N row col role+byte<
      (a (region r role)).base+(a (region r role)).extent*(a (region r role)).width ∧
    nativeAddress a r stride M N row col role+byte<2^64 := by
  obtain ⟨ha,hl⟩ := (machine_guard_iff hadmit hc f role).mp hg
  have hoff := NativeAccess.all_roles_bounded f r stride M N row col _ (h.accesses row hr) role ha hl
  have hmul := Nat.mul_le_mul_right (a (region r role)).width (Nat.succ_le_of_lt hoff)
  have hfinite := (a (region r role)).bytes_finite
  rw [native_address_exact f r stride M N block a hadmit h row col hr hc role hg]
  simp only [layout,Address.byteAddress]
  rw [Nat.succ_mul] at hmul
  omega

#axiomsClean offset_exact
#axiomsClean byte_address_exact
#axiomsClean native_address_exact
#axiomsClean native_access_span
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
