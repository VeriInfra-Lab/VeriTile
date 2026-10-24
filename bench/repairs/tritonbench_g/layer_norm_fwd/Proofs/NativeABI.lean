import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeAccess

/-! Selected CUDA launch/metadata word interface. The launch limit is an
ordinary device API admission condition, not a new wrapper assertion. The
wrapper still forwards every M, including zero and oversized grids, to dispatch.
This module proves the word/mask/access equations; connecting its role-based
operations to the whole staged schedule is separate work. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeABI
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def signed32 (n : Nat) : Int :=
  if n % 2^32 < 2^31 then (n % 2^32 : Nat) else (n % 2^32 : Nat) - (2^32 : Int)

theorem signed32_exact {n : Nat} (h : n<2^31) : signed32 n=(n : Int) := by
  have hn : n<2^32 := lt_trans h (by norm_num)
  simp only [signed32,Nat.mod_eq_of_lt hn,if_pos h]

/-- The source casts signed program_id i32 to i64 before multiplying. -/
def signExtend32 (n : Nat) : Nat := (BitVec.ofInt 64 (signed32 n)).toNat

/-- The compiler may use unsigned extension for known nonnegative CTA ids. -/
def zeroExtend32 (n : Nat) : Nat := n % 2^32

theorem signExtend32_exact {n : Nat} (h : n<2^31) : signExtend32 n=n := by
  have hn : n<2^64 := lt_trans h (by norm_num)
  simp only [signExtend32,signed32_exact h,BitVec.ofInt_natCast,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hn]

theorem zeroExtend32_exact {n : Nat} (h : n<2^31) : zeroExtend32 n=n :=
  Nat.mod_eq_of_lt (lt_trans h (by norm_num))

/-- Triton 3.7.1 runtime integer ABI for nonnegative tensor metadata. The
u64 branch occurs only outside the ordinary signed-i64 metadata domain. -/
inductive ScalarABI where | i32 | i64 | u64 deriving DecidableEq

def scalarABI (n : Nat) : ScalarABI :=
  if n<2^31 then .i32 else if n<2^63 then .i64 else .u64

def metadataWord (n : Nat) : Nat :=
  match scalarABI n with | .i32 => signExtend32 n | .i64 | .u64 => wrap64 n

theorem metadata_word_exact {n : Nat} (h : n<2^63) : metadataWord n=n := by
  by_cases hn : n<2^31
  · simp only [metadataWord,scalarABI,if_pos hn,signExtend32_exact hn]
  · simp only [metadataWord,scalarABI,if_neg hn,if_pos h,wrap64_exact h]

structure Accepted (M N block : Nat) : Prop where
  gridX : M≤2^31-1
  width : N≤block
  blockPositive : 0<block
  blockLimit : block≤16384

inductive AdmissionFailure where | gridLimit | blockConfiguration deriving DecidableEq

/-- A model of the ordinary selected-device admission boundary, explicitly
retaining failure instead of silently restricting the wrapper's input domain.
Other allocation/compilation/device failures are retained by Wrapper.Local. -/
def admit (M N block : Nat) : Except AdmissionFailure {u : Unit // Accepted M N block} :=
  if hg : M≤2^31-1 then
    if hb : N≤block ∧ 0<block ∧ block≤16384 then .ok ⟨(),hg,hb.1,hb.2.1,hb.2.2⟩
    else .error .blockConfiguration
  else .error .gridLimit

theorem oversized_grid_rejected (M N block : Nat) (h : 2^31-1<M) :
    admit M N block=.error .gridLimit := by simp only [admit,dif_neg (show ¬M≤2^31-1 by omega)]

theorem accepted_row {M N block row : Nat} (h : Accepted M N block) (hr : row<M) : row<2^31 := by
  have := h.gridX
  omega

theorem accepted_col {M N block col : Nat} (h : Accepted M N block) (hc : col<block) : col<2^31 := by
  have := h.blockLimit
  omega

theorem accepted_N {M N block : Nat} (h : Accepted M N block) : N<2^31 := by
  have := h.width
  have := h.blockLimit
  omega

theorem program_id_casts_agree {M N block row : Nat} (h : Accepted M N block) (hr : row<M) :
    signExtend32 row=row ∧ zeroExtend32 row=row :=
  ⟨signExtend32_exact (accepted_row h hr),zeroExtend32_exact (accepted_row h hr)⟩

def signedMask (N col : Nat) : Bool := decide (signed32 col<signed32 N)

theorem signed_mask_exact {M N block col : Nat} (h : Accepted M N block) (hc : col<block) :
    signedMask N col=decide (col<N) := by
  simp [signedMask,signed32_exact (accepted_col h hc),signed32_exact (accepted_N h)]

def scalarRole : Role → Bool
  | .mean | .rstd | .rowscale | .rowscale1 | .seeds | .seeds1 => Bool.true
  | _ => Bool.false

def machineGuard (f : Flags) (N col : Nat) (role : Role) : Bool :=
  decide (active f role) && (if scalarRole role then Bool.true else signedMask N col)

theorem machine_guard_iff {M N block col : Nat} (h : Accepted M N block) (hc : col<block)
    (f : Flags) (role : Role) : machineGuard f N col role=Bool.true ↔
      active f role ∧ NativeAccess.laneActive N col role := by
  unfold machineGuard
  rw [signed_mask_exact h hc]
  cases role <;> simp [scalarRole,NativeAccess.laneActive]

def strideWords (s : Strides) : Strides where
  x := metadataWord s.x
  y := metadataWord s.y
  residual := metadataWord s.residual
  residualOut := metadataWord s.residualOut
  x1 := metadataWord s.x1
  y1 := metadataWord s.y1

theorem stride_words_exact (s : Strides) (M N : Nat) (h : Metadata s M N) : strideWords s=s := by
  cases s
  simp only [strideWords,metadata_word_exact h.x,metadata_word_exact h.y,
    metadata_word_exact h.residual,metadata_word_exact h.residualOut,
    metadata_word_exact h.x1,metadata_word_exact h.y1]

/-- Byte pointer expression after parameter loading, source pid extension and
lane extension, followed by the source's nested fixed-width arithmetic. -/
def nativeAddress (a : RegionName → Allocation) (r : Regions) (s : Strides)
    (M N row col : Nat) (role : Role) : Nat :=
  machineByteAddress a r (strideWords s) (metadataWord M) (metadataWord N)
    (signExtend32 row) (zeroExtend32 col) role

theorem native_address_exact (f : Flags) (r : Regions) (s : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (hadmit : Accepted M N block)
    (hd : PhysicalDomain.Domain f r s M N a) (row col : Nat) (hr : row<M) (hc : col<block)
    (role : Role) (hg : machineGuard f N col role=Bool.true) :
    nativeAddress a r s M N row col role=(layout a).address (region r role,offset s M N row col role) := by
  obtain ⟨ha,hl⟩ := (machine_guard_iff hadmit hc f role).mp hg
  simp only [nativeAddress,stride_words_exact s M N hd.metadata,
    metadata_word_exact hd.metadata.rows,metadata_word_exact hd.metadata.cols,
    signExtend32_exact (accepted_row hadmit hr),zeroExtend32_exact (accepted_col hadmit hc)]
  exact NativeAccess.all_machine_addresses f r s M N a hd row col hr role ha hl

/-- Every byte in the selected 1/4/8-byte access lies in the allocation and
below 2^64, rather than proving only that the first byte is representable. -/
theorem native_access_span (f : Flags) (r : Regions) (s : Strides) (M N block : Nat)
    (a : RegionName → Allocation) (hadmit : Accepted M N block)
    (hd : PhysicalDomain.Domain f r s M N a) (row col : Nat) (hr : row<M) (hc : col<block)
    (role : Role) (hg : machineGuard f N col role=Bool.true) (byte : Nat) (hb : byte<roleWidth role) :
    (a (region r role)).base≤nativeAddress a r s M N row col role+byte ∧
    nativeAddress a r s M N row col role+byte<
      (a (region r role)).base+(a (region r role)).extent*(a (region r role)).width ∧
    nativeAddress a r s M N row col role+byte<2^64 := by
  obtain ⟨ha,hl⟩ := (machine_guard_iff hadmit hc f role).mp hg
  have hoff := NativeAccess.all_roles_bounded f r s M N row col _
    (hd.accesses row hr) role ha hl
  have hmul := Nat.mul_le_mul_right (a (region r role)).width (Nat.succ_le_of_lt hoff)
  have hwidth := hd.widths role ha
  have hfinite := (a (region r role)).bytes_finite
  rw [native_address_exact f r s M N block a hadmit hd row col hr hc role hg]
  change (a (region r role)).base≤byteAddress (a (region r role)) (offset s M N row col role)+byte ∧ _
  simp only [layout,byteAddress]
  rw [Nat.succ_mul] at hmul
  omega

#axiomsClean metadata_word_exact
#axiomsClean program_id_casts_agree
#axiomsClean signed_mask_exact
#axiomsClean machine_guard_iff
#axiomsClean native_address_exact
#axiomsClean native_access_span
#axiomsClean oversized_grid_rejected
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeABI
