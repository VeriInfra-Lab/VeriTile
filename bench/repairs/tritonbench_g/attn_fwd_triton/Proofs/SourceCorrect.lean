import bench.repairs.tritonbench_g.attn_fwd_triton.Kernels
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostWrapper
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostEmpty
import VeriTile.Meta.StatementAudit

/-! Public real target and proved structural/mathematical components.

Intended host domain: logical FP32 Q/K/V with the same [B,H,N,D] shape, FP32
packed-flat Q/K scales of length at least B*H*ceil(N/128), B*H*ceil(N/64), all
on the same device. Arbitrary valid independent nonnegative strides, storage
offsets, read-only aliases, zero strides and lazy-negative views are allowed.
resolve_neg preserves every logical coordinate; scales are then materialized
in logical flattened order. Empty outputs return without a launch.

Host grid is [ceil(N/128), B*H, 1], BM=128, BN=64,
BD=max(16,nextPowerOfTwo(D)), STAGE=3, eight warps and three stages. Output is a
fresh BF16 empty_like allocation, addressed using its own actual strides.
No supplied output is copied or replaced. Q/K/V are read only. There is no
scratch buffer after removing the original unused M allocation.

The independent target below is the exact causal normalized base-2 weighted
sum over all D input channels. Real casts are erased. Production FP16 casts,
FP16 value-dot, FP32 operations and BF16 output remain in Kernels/FPEquiv.

The complete translated source now has a whole-grid Real theorem below.
Its assumptions are ordinary valid views, fresh output bytes, nonoverlapping
output layout, and coordinate-preserving tensor-library preparation. Read
inputs may alias. Physical memory is modeled as typed cells indexed by byte
starts; raw-byte codecs, compiler/GPU conformance, and production numerical
accuracy are not asserted by this Real theorem. CUDA setup feasibility and
its exact native source consequences are consumed by the same-request public
return theorem. Impossible launches are ordinary API outcomes. Empty returns
need no scale-length or launch premise, including D=0 with nonempty B/H/N. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Correctness
open VeriTile.Triton
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
open scoped VeriTile.Spec
noncomputable section

abbrev Inputs := RealProofs.Inputs
abbrev Parameters := RealProofs.Parameters
abbrev expected := RealProofs.expected
abbrev source := RealProofs.source

/-- Public IO coordinates, independently for each allocation. -/
def queryAddress (p : Parameters) (z h m d : Nat) :=
  RealProofs.address p.queryBatch p.queryHead p.queryRow p.queryDim z h m d

def keyAddress (p : Parameters) (z h n d : Nat) :=
  RealProofs.address p.keyBatch p.keyHead p.keyRow p.keyDim z h n d

def valueAddress (p : Parameters) (z h n d : Nat) :=
  RealProofs.address p.valueBatch p.valueHead p.valueRow p.valueDim z h n d

def outputAddress (p : Parameters) (z h m d : Nat) :=
  RealProofs.address p.outputBatch p.outputHead p.outputRow p.outputDim z h m d

def outputWindow (p : Parameters) (offset : Nat) : Prop :=
  ∃ z < p.batches, ∃ h < p.heads, ∃ m < p.context, ∃ d < p.headDim,
    outputAddress p z h m d = offset

/-- Every logical output is read from the actual fresh physical output view;
the full typed-cell complement is preserved, and every grid trace is bounded. -/
specification whole_grid_real (a : Layout) (A : FlatAlloc) (p : Parameters) (BD : Nat)
    (base : Role → Nat) (views : Views a A p base)
    (initial : AliasMemory.Physical a.toLayout) (template : BlockState)
    (xs : Inputs) (agree : LogicalAgreement.Agree p (Host.logicalInputs a A p initial template) xs)
    (bounds : RegionBounds) (windows : ViewBounds.Windows bounds (Relocation.parameters p A.flat) (Relocation.offsets A p))
    (hD : p.headDim≤BD) (hstage : p.stage=3) (reciprocal : Bool)
    (nonoverlap : OutputLayout.NonOverlapping p)
    (fresh : FreshBytes a (GridFootprint.window A p)) :
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p BD reciprocal) (GridCoordinates.grid p)
        (AliasMemory.pull a.toLayout initial template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p base views i)) =
          expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val xs) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList bounds (GridLaunch.kernel A p BD reciprocal).body
          ((AliasMemory.pull a.toLayout initial template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=initial cell) := by
  exact Host.launch a A p BD base views initial template xs agree bounds windows
    hD hstage reciprocal nonoverlap fresh

open HostWrapper
open HostSetup (Call calls beforeEmpty nonemptyChecks)
open HostShape (blockD)

specification whole_public_real (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : p.batches*p.heads*p.context*p.headDim≠0)
    (hlaunch : NativeControl.CudaLaunchable p (blockD p.headDim)) (reciprocal : Bool) :
    ret.returnedOutput=outputDescriptor p ret.base ∧
    HostSetup.run p (blockD p.headDim) ok=.launch hne hlaunch ∧
    NativeCertificate p a A caller template xs ret ∧
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p (blockD p.headDim) reciprocal) (GridCoordinates.grid p)
        (AliasMemory.pull a.toLayout ret.initial template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p ret.base ret.views i))=
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val xs) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList ret.bounds (GridLaunch.kernel A p (blockD p.headDim) reciprocal).body
          ((AliasMemory.pull a.toLayout ret.initial template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.owned cell → final cell=caller cell) := by
  exact HostWrapper.successful_return p a A caller template xs ret ok hbefore hscales hcompile hne hlaunch reciprocal

abbrev empty_public_return := @HostEmpty.empty_return
abbrev infeasible_launch_failure := @HostSetup.infeasible_launch_fails
abbrev ordered_first_failure := @HostSetup.first_failure

/-- Active native element/byte addresses follow from ordinary allocation and
view extents. Dead pointer advances may wrap modulo 2^64. -/
abbrev native_active_addresses := @NativeOffsets.all_roles
abbrev native_key_helper_addresses := @NativeSource.key_active
abbrev native_value_helper_addresses := @NativeSource.value_active
abbrev native_query_scale_addresses := @NativeSource.query_scale_active
abbrev native_key_scale_addresses := @NativeSource.key_scale_active
abbrev native_cdiv_control := @NativeControl.cdiv_inputs
abbrev native_row_control := @NativeControl.row_arithmetic
abbrev native_key_control := @NativeControl.key_arithmetic
abbrev empty_output := @Host.empty_output

theorem every_row_has_one_grid_coordinate (N m : Nat) (hm : m < N) :
    m / 128 < (N+127)/128 ∧ m % 128 < 128 ∧
      (m / 128)*128+m%128 = m := RealProofs.query_grid_coverage N m hm

theorem active_grid_coordinates_are_unique (m pid lane : Nat) (hl : lane < 128)
    (he : pid*128+lane=m) : pid=m/128 ∧ lane=m%128 :=
  RealProofs.query_grid_unique m pid lane hl he

theorem every_head_has_a_grid_coordinate (B H z h : Nat) (hz : z < B) (hh : h < H) :
    z*H+h < B*H ∧ (z*H+h)/H = z ∧ (z*H+h)%H = h :=
  ⟨RealProofs.head_grid_coverage B H z h hz hh, RealProofs.head_grid_decode H z h hh⟩

theorem every_channel_is_in_store_mask (D BD d : Nat) (hD : D ≤ BD) (hd : d < D) :
    d < BD ∧ decide (d < D) = Bool.true := RealProofs.complete_head_mask D BD d hD hd

theorem every_visited_key_scale_is_in_packed_row (N start : Nat) (hs : start < N) :
    start/64 < (N+63)/64 := RealProofs.scale_tail_bound N start hs

/-- Reuses the existing pure online normalization proof for all input sizes.
The partition premise is only about lists of mathematical inputs; it is not
an output, execution, compiler, memory or protocol assumption. -/
theorem independent_formula_online (H N D z h m d : Nat) (xs : Inputs)
    (blocks : List (List (ℝ × ℝ)))
    (partition : blocks.flatten = RealProofs.pairs H N D z h m d xs) :
    let state := blocks.foldl
      VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula.summarize
      VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula.initial
    state.numerator / state.denominator = expected H N D z h m d xs :=
  RealProofs.online_formula H N D z h m d xs blocks partition

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Correctness
