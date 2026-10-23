import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.AliasMemory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Address

namespace VeriTile.Bench.Repairs.LayerNormFwd.PhysicalDomain
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open AliasMemory
noncomputable section

/-- Byte-addressed typed cells. Each view retains its own storage-offset base;
all elements of every allocated view are live. Read views need not be disjoint.
This uses the same ordinary tensor-library allocation metadata as Address. -/
def layout (a : RegionName → Address.Allocation) : Layout where
  capacity := 2^64
  address := fun k => Address.byteAddress (a k.1) k.2
  live := fun k => k.2 < (a k.1).extent
  bounded := fun k hk => (Address.byte_address (a k.1) hk).2.2.1

/-- Ordinary signed tensor metadata; these are not i32-product guards. -/
structure Metadata (stride : Strides) (M N : Nat) : Prop where
  rows : M < 2^63
  cols : N < 2^63
  x : stride.x < 2^63
  y : stride.y < 2^63
  residual : stride.residual < 2^63
  residualOut : stride.residualOut < 2^63
  x1 : stride.x1 < 2^63
  y1 : stride.y1 < 2^63

def roleWidth : Address.Role → Nat
  | .seeds | .seeds1 => 8
  | .mask | .mask1 => 1
  | _ => 4

def machineOffset (s : Strides) (M N row col : Nat) : Address.Role → Nat
  | .x => Address.matrixOffset64 row s.x col
  | .y => Address.matrixOffset64 row s.y col
  | .residual => Address.matrixOffset64 row s.residual col
  | .residualOut => Address.matrixOffset64 row s.residualOut col
  | .x1 => Address.matrixOffset64 row s.x1 col
  | .y1 => Address.matrixOffset64 row s.y1 col
  | .weight | .bias | .weight1 | .bias1 => Address.wrap64 col
  | .mean | .rstd | .rowscale | .seeds => Address.wrap64 row
  | .rowscale1 | .seeds1 => Address.add64 (Address.wrap64 M) (Address.wrap64 row)
  | .mask => Address.matrixOffset64 row N col
  | .mask1 => Address.maskOffset64 M row N col

/-- Ordinary allocation coverage plus fresh output storage, with no
read/read separation. Byte codecs are not numerical/output assumptions. -/
structure Domain (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) : Prop where
  metadata : Metadata stride M N
  widths : ∀ role, Address.active f role → (a (Address.region r role)).width=roleWidth role
  accesses : ∀ row, row<M → FlatDomain.RowBounds f r stride M N row (fun q => (a q).extent)
  fresh : FreshWrites (layout a) (Writes f r stride M N)

/-- Ordinary nonempty-span bounds imply the exact access-domain contract.
The converse is deliberately absent: N=0 vector accesses require no storage. -/
theorem Domain.of_extents (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (hm : Metadata stride M N)
    (hw : ∀ role, Address.active f role → (a (Address.region r role)).width=roleWidth role)
    (he : 0<M → FlatDomain.Extents f r stride M N (fun q => (a q).extent))
    (hf : FreshWrites (layout a) (Writes f r stride M N)) : Domain f r stride M N a :=
  ⟨hm,hw,fun row hr => (he (by omega)).rowBounds f r stride M N row _ hr,hf⟩

/-- An empty row grid performs no accesses and therefore requires no input
or output span. In particular the conventional `(M-1)*stride+N` formula must
not demand N live cells from an ordinary empty tensor when M=0. -/
theorem zero_rows_domain (f : Flags) (r : Regions) (stride : Strides) (N : Nat)
    (a : RegionName → Address.Allocation) (hm : Metadata stride 0 N)
    (hw : ∀ role, Address.active f role → (a (Address.region r role)).width=roleWidth role) :
    Domain f r stride 0 N a := by
  refine ⟨hm,hw,?_,?_⟩
  · intro row h; omega
  · intro k hk
    obtain ⟨row,hr,_⟩ := hk
    omega

/-- Only the scalar paths remain for an empty feature dimension. In
particular every matrix allocation may have extent zero at any row stride. -/
structure EmptyColumnsScalars (f : Flags) (r : Regions) (M : Nat)
    (a : RegionName → Address.Allocation) : Prop where
  rstd : M≤(a r.rstd).extent
  mean : f.rms=Bool.false → M≤(a r.mean).extent
  rowscale : f.rowscale=Bool.true → M≤(a r.rowscale).extent
  rowscale1 : f.x1=Bool.true → f.rowscale=Bool.true → 2*M≤(a r.rowscale).extent
  seeds : f.dropout=Bool.true → M≤(a r.seeds).extent
  seeds1 : f.x1=Bool.true → f.dropout=Bool.true → 2*M≤(a r.seeds).extent

theorem zero_columns_domain (f : Flags) (r : Regions) (stride : Strides) (M : Nat)
    (a : RegionName → Address.Allocation) (hm : Metadata stride M 0)
    (hw : ∀ role, Address.active f role → (a (Address.region r role)).width=roleWidth role)
    (hs : EmptyColumnsScalars f r M a)
    (hf : FreshWrites (layout a) (Writes f r stride M 0)) : Domain f r stride M 0 a := by
  refine ⟨hm,hw,?_,hf⟩
  intro row hr
  refine ⟨?_,?_,?_,lt_of_lt_of_le hr hs.rstd,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals first
    | (exact fun h => lt_of_lt_of_le hr (hs.mean h))
    | (exact fun h => lt_of_lt_of_le hr (hs.rowscale h))
    | (exact fun hx hs' => lt_of_lt_of_le (Domain.second_row_lt hr) (hs.rowscale1 hx hs'))
    | (exact fun h => lt_of_lt_of_le hr (hs.seeds h))
    | (exact fun hx hd => lt_of_lt_of_le (Domain.second_row_lt hr) (hs.seeds1 hx hd))
    | (intros; omega)

theorem all_accesses_live (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<N)
    (role : Address.Role) (ha : Address.active f role) :
    (layout a).live (Address.region r role,Address.offset stride M N row col role) :=
  Address.all_roles_bounded f r stride M N row col _
    (h.accesses row hr) hc role ha

/-- Every actual input and output role reads its mapped physical cell;
no active input can use the out-of-view default in AliasMemory.read. -/
theorem active_read (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (h : Domain f r stride M N a)
    (physical : Physical (layout a)) (row col : Nat) (hr : row<M) (hc : col<N)
    (role : Address.Role) (ha : Address.active f role) :
    read (layout a) physical (Address.region r role,Address.offset stride M N row col role) =
      physical ⟨Address.byteAddress (a (Address.region r role)) (Address.offset stride M N row col role),
        (layout a).bounded _ (all_accesses_live f r stride M N a h row col hr hc role ha)⟩ := by
  exact dif_pos (all_accesses_live f r stride M N a h row col hr hc role ha)

theorem all_accesses_finite (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<N)
    (role : Address.Role) (ha : Address.active f role) :
    Address.signed64 (Address.offset stride M N row col role) =
      (Address.offset stride M N row col role : Int) ∧
    Address.byteAddress (a (Address.region r role)) (Address.offset stride M N row col role)<2^64 := by
  have hb := all_accesses_live f r stride M N a h row col hr hc role ha
  have he := Address.byte_address (a (Address.region r role)) hb
  exact ⟨he.2.2.2,he.2.2.1⟩

/-- The actual nested fixed-width offset expression is exact for every active
read/write role. Allocation coverage, signed metadata and the source dtype
widths compose with the physical view mapping. -/
theorem all_machine_addresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (h : Domain f r stride M N a)
    (row col : Nat) (hr : row<M) (hc : col<N)
    (role : Address.Role) (ha : Address.active f role) :
    machineOffset stride M N row col role=Address.offset stride M N row col role ∧
    (layout a).address (Address.region r role,Address.offset stride M N row col role) =
      (a (Address.region r role)).base + machineOffset stride M N row col role * roleWidth role := by
  have hb := all_accesses_live f r stride M N a h row col hr hc role ha
  change Address.offset stride M N row col role < (a (Address.region r role)).extent at hb
  have he := (a (Address.region r role)).signed_extent
  have hrow : row<2^63 := lt_trans hr h.metadata.rows
  have hcol : col<2^63 := lt_trans hc h.metadata.cols
  have exactOffset : machineOffset stride M N row col role=Address.offset stride M N row col role := by
    cases role with
    | x => exact Address.matrix_machine_exact hrow h.metadata.x hb he
    | y => exact Address.matrix_machine_exact hrow h.metadata.y hb he
    | residual => exact Address.matrix_machine_exact hrow h.metadata.residual hb he
    | residualOut => exact Address.matrix_machine_exact hrow h.metadata.residualOut hb he
    | x1 => exact Address.matrix_machine_exact hrow h.metadata.x1 hb he
    | y1 => exact Address.matrix_machine_exact hrow h.metadata.y1 hb he
    | mask => exact Address.matrix_machine_exact hrow h.metadata.cols hb he
    | mask1 => exact Address.mask_machine_exact (by omega) h.metadata.cols hb he
    | weight | bias | weight1 | bias1 => exact Address.wrap64_exact hcol
    | mean | rstd | rowscale | seeds => exact Address.wrap64_exact hrow
    | rowscale1 | seeds1 =>
      have hh : M+row<2^63 := lt_of_lt_of_le hb he
      simp only [machineOffset,Address.offset,Address.add64,Address.wrap64_exact h.metadata.rows,
        Address.wrap64_exact hrow,Address.wrap64_exact hh]
  refine ⟨exactOffset,?_⟩
  simp only [layout,Address.byteAddress,exactOffset,h.widths role ha]

def machineByteAddress (a : RegionName → Address.Allocation) (r : Regions)
    (stride : Strides) (M N row col : Nat) (role : Address.Role) : Nat :=
  Address.add64 (Address.wrap64 (a (Address.region r role)).base)
    (Address.mul64 (machineOffset stride M N row col role) (roleWidth role))

def MachineAddresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) : Prop :=
  ∀ row col, row<M → col<N → ∀ role, Address.active f role →
    machineByteAddress a r stride M N row col role =
      (layout a).address (Address.region r role,Address.offset stride M N row col role)

/-- Includes both actual byte multiplication and pointer-base addition at
unsigned 64-bit width; high pointer bases need not fit signed i64. -/
theorem machine_addresses (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (a : RegionName → Address.Allocation) (h : Domain f r stride M N a) :
    MachineAddresses f r stride M N a := by
  intro row col hr hc role ha
  have he := (all_machine_addresses f r stride M N a h row col hr hc role ha).1
  have hb := (all_accesses_finite f r stride M N a h row col hr hc role ha).2
  change (a (Address.region r role)).base + Address.offset stride M N row col role *
    (a (Address.region r role)).width < 2^64 at hb
  have hbase : (a (Address.region r role)).base<2^64 := by omega
  have hmul : Address.offset stride M N row col role * (a (Address.region r role)).width<2^64 := by omega
  simp only [machineByteAddress,he,←h.widths role ha,Address.add64,Address.mul64,
    Address.wrap64,Nat.mod_eq_of_lt hbase,Nat.mod_eq_of_lt hmul,Nat.mod_eq_of_lt hb,
    layout,Address.byteAddress]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.PhysicalDomain
