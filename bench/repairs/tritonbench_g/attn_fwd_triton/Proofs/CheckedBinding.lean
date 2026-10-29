import bench.repairs.tritonbench_g.attn_fwd_triton.Contract
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
set_option maxHeartbeats 1000000

def realGeometry (p : RealProofs.Parameters) : Geometry where
  batches:=p.batches
  heads:=p.heads
  context:=p.context
  headDim:=p.headDim
  stride
    | .query=>⟨p.queryBatch,p.queryHead,p.queryRow,p.queryDim⟩
    | .key=>⟨p.keyBatch,p.keyHead,p.keyRow,p.keyDim⟩
    | .value=>⟨p.valueBatch,p.valueHead,p.valueRow,p.valueDim⟩
    | .output=>⟨p.outputBatch,p.outputHead,p.outputRow,p.outputDim⟩
    | _=>⟨0,0,1,0⟩

/-- Ordinary backed tensor spans and a finite flat-storage interpretation.
Scale-length coverage is derived from checks, rather than supplied here. -/
structure Storage (p : RealProofs.Parameters) (A : FlatAlloc) (bounds : RegionBounds)
    (qs ks : TensorMetadata) where
  capacity : Role→Nat
  matrix : ∀ r, r≠Role.queryScale → r≠Role.keyScale → span (realGeometry p) r≤capacity r
  queryScale : qs.numel≤capacity .queryScale
  keyScale : ks.numel≤capacity .keyScale
  flat : ∀ r, A.base (region p r)+capacity r≤bounds A.flat

theorem checked_windows (p : RealProofs.Parameters) (A : FlatAlloc) (bounds : RegionBounds)
    (qs ks : TensorMetadata) (storage : Storage p A bounds qs ks)
    (hl : launchChecks (realGeometry p) qs ks=Bool.true) :
    ViewBounds.Windows bounds (Relocation.parameters p A.flat) (Relocation.offsets A p) := by
  have matrix (r : Role) (hqs : r≠Role.queryScale) (hks : r≠Role.keyScale)
      (z h n d : Nat) (hz : z<p.batches) (hh : h<p.heads) (hn : n<p.context) (hd : d<p.headDim) :
      ((realGeometry p).stride r).at z h n d+A.base (region p r)<bounds A.flat := by
    have hb:=active_bound (realGeometry p) r hqs hks z h n d hz hh hn hd
    have hs:=storage.matrix r hqs hks
    have hf:=storage.flat r
    omega
  constructor
  · exact fun z hz h hh n hn d hd=>matrix .query (by decide) (by decide) z h n d hz hh hn hd
  · exact fun z hz h hh n hn d hd=>matrix .key (by decide) (by decide) z h n d hz hh hn hd
  · exact fun z hz h hh n hn d hd=>matrix .value (by decide) (by decide) z h n d hz hh hn hd
  · exact fun z hz h hh n hn d hd=>matrix .output (by decide) (by decide) z h n d hz hh hn hd
  · simp only [launchChecks,decide_eq_true_eq] at hl
    have hs:=storage.queryScale
    have hf:=storage.flat .queryScale
    change span (realGeometry p) .queryScale+A.base p.QS≤bounds A.flat
    change A.base p.QS+storage.capacity .queryScale≤bounds A.flat at hf
    omega
  · simp only [launchChecks,decide_eq_true_eq] at hl
    have hs:=storage.keyScale
    have hf:=storage.flat .keyScale
    change span (realGeometry p) .keyScale+A.base p.KS≤bounds A.flat
    change A.base p.KS+storage.capacity .keyScale≤bounds A.flat at hf
    omega

/-- Structural safety of the complete two-loop source, independent of any
numerical correspondence. The ordinary tensor backing is enough after checks. -/
theorem trace_safe (p : RealProofs.Parameters) (A : FlatAlloc) (bounds : RegionBounds)
    (qs ks : TensorMetadata) (storage : Storage p A bounds qs ks)
    (hl : launchChecks (realGeometry p) qs ks=Bool.true) (s : BlockState)
    (idx : GridIndex (GridCoordinates.grid p)) :
    Stmt.TraceSafeList bounds (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false).body
      (s.withGridIndex idx) :=
  GridLaunch.safe A bounds s p _ Bool.false (checked_windows p A bounds qs ks storage hl) idx

#axiomsClean checked_windows
#axiomsClean trace_safe
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
