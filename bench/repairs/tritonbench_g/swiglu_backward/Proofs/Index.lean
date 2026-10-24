import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Binding

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section
set_option maxHeartbeats 2000000

theorem checked_lane_correspondence (p : Prepared α call entry preexisting) (h : LaunchContract p byteBase) (a : Access) (on : a.active call.recompute)
    (config : Config) (r b lane : Nat) (hr : r < p.batchShape.prod)
    (hb : b < (p.ncols+config.block-1)/config.block) (hl : lane < config.block) :
    LaneCorrespondence p a config r b lane := by
  have columnBound := padded_column p.ncols config b lane hb hl
  have rowBound := Nat.mul_le_mul_right (a.view p).rowStride (Nat.le_of_lt hr)
  have total := h.padded32 a on
  have hrow : r*(a.view p).rowStride < signedLimit := by dsimp [signedLimit]; omega
  have hstart : b*config.block < signedLimit := by dsimp [signedLimit]; omega
  have hcol : b*config.block+lane < signedLimit := by dsimp [signedLimit]; omega
  have htotal : r*(a.view p).rowStride+(b*config.block+lane) < signedLimit := by dsimp [signedLimit]; omega
  have hc : (cols32 b config.block lane).toInt = (b*config.block+lane : Nat) := by
    simp only [cols32,start32,SwigluFwd.Whole.cols32,SwigluFwd.Whole.start32,
      ←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact SwigluFwd.Whole.signed_ofNat _ hcol
  refine ⟨?_,?_,hc,?_,?_⟩
  · simpa only [row32,SwigluFwd.Whole.row32,←BitVec.ofNat_mul] using SwigluFwd.Whole.signed_ofNat _ hrow
  · simpa only [start32,SwigluFwd.Whole.start32,←BitVec.ofNat_mul] using SwigluFwd.Whole.signed_ofNat _ hstart
  · simpa only [row32,cols32,start32,SwigluFwd.Whole.row32,SwigluFwd.Whole.cols32,
      SwigluFwd.Whole.start32,←BitVec.ofNat_mul,←BitVec.ofNat_add] using SwigluFwd.Whole.signed_ofNat _ htotal
  · change decide ((cols32 b config.block lane).toInt < (BitVec.ofNat 32 p.ncols).toInt) = _
    rw [hc,SwigluFwd.Whole.signed_ofNat _ h.columns32]
    simp only [Int.ofNat_lt]

/-- Source indices cover all masked lanes; physical address and alignment
claims apply to actual accesses. No padded allocation is manufactured. -/
def CheckedIndices (p : Prepared α call entry preexisting) (s : Storage p) : Prop :=
  ∀ a : Access, a.active call.recompute → ∀ config r b lane,
    r < p.batchShape.prod → b < (p.ncols+config.block-1)/config.block → lane < config.block →
    LaneCorrespondence p a config r b lane ∧
    (b*config.block+lane < p.ncols →
      let address := (pointer64 (s.byteBase (a.view p).allocation) (a.view p).base
        (row32 r (a.view p).rowStride) (cols32 b config.block lane)).toNat
      address = s.byteBase (a.view p).allocation+4*((a.view p).base+r*(a.view p).rowStride+(b*config.block+lane)) ∧
      (a.view p).base+r*(a.view p).rowStride+(b*config.block+lane) < s.capacity (a.view p).allocation ∧
      address % 4 = 0)

theorem checked_indices (p : Prepared α call entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase) : CheckedIndices p s := by
  intro a on config r b lane hr hb hl
  have indices := checked_lane_correspondence p h a on config r b lane hr hb hl
  refine ⟨indices,?_⟩
  intro active
  have bound := access_bound p (checkedDomain p s hc h) a on r (b*config.block+lane) hr active
  have allocation := s.allocation64 (a.view p).allocation
  have address := SwigluFwd.Whole.pointer64_exact (s.byteBase (a.view p).allocation) (a.view p).base
    (r*(a.view p).rowStride) (b*config.block+lane) _ _ indices.row indices.column
    (by change (a.view p).base+r*(a.view p).rowStride+(b*config.block+lane) < s.capacity (a.view p).allocation at bound; dsimp [SwigluFwd.Whole.addressLimit]; omega)
  have aligned := h.alignment (Nat.zero_lt_of_lt hr) a on
  exact ⟨address,bound,by rw [address]; omega⟩

#axiomsClean checked_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward
