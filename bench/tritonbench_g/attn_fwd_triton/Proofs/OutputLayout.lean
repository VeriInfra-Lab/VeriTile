import bench.tritonbench_g.attn_fwd_triton.Proofs.ViewBounds
import bench.tritonbench_g.attn_fwd_triton.Proofs.Output

/-! The fresh output tensor is a nonoverlapping logical view. This ordinary
layout property discharges the per-tile readback injectivity premise. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.OutputLayout
open VeriTile.Triton Memory
noncomputable section

abbrev Coordinate (p : Parameters) := Fin p.batches × Fin p.heads × Fin p.context × Fin p.headDim

def address (p : Parameters) (i : Coordinate p) : Nat :=
  RealProofs.address p.outputBatch p.outputHead p.outputRow p.outputDim
    i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val

def NonOverlapping (p : Parameters) : Prop := Function.Injective (address p)

theorem active_bounds (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD])
    (hi : Output.active c BM BD i = Bool.true) :
    c.startM*BM+i.1.val < c.p.context ∧ i.2.1.val < c.p.headDim := by
  simpa [Output.active,Output.outputConfig,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.active] using hi

def coordinate (c : Config) (BM BD : Nat) (hp : ViewBounds.Program c)
    (i : TileIndex [BM,BD]) (hi : Output.active c BM BD i = Bool.true) : Coordinate c.p :=
  (⟨c.z,hp.batch⟩,⟨c.h,hp.head⟩,⟨c.startM*BM+i.1.val,(active_bounds c BM BD i hi).1⟩,
    ⟨i.2.1.val,(active_bounds c BM BD i hi).2⟩)

theorem address_coordinate (c : Config) (BM BD : Nat) (hp : ViewBounds.Program c)
    (i : TileIndex [BM,BD]) (hi : Output.active c BM BD i = Bool.true) :
    Output.address c BM BD i = address c.p (coordinate c BM BD hp i hi)+c.offsets.output := by
  simp [Output.address,Output.outputConfig,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealEpilogue.address,
    address,coordinate,RealProofs.address,obase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem active_injective (c : Config) (BM BD : Nat) (hp : ViewBounds.Program c)
    (h : NonOverlapping c.p) :
    ∀ x y, Output.active c BM BD x = Bool.true → Output.active c BM BD y = Bool.true →
      Output.address c BM BD x = Output.address c BM BD y → x=y := by
  intro x y hx hy he
  rw [address_coordinate c BM BD hp x hx,address_coordinate c BM BD hp y hy] at he
  have hh := h (Nat.add_right_cancel he)
  have hr := congrArg (fun i : Coordinate c.p => i.2.2.1.val) hh
  have hd := congrArg (fun i : Coordinate c.p => i.2.2.2.val) hh
  apply Prod.ext
  · apply Fin.ext
    exact Nat.add_left_cancel hr
  · apply Prod.ext
    · exact Fin.ext hd
    · exact Subsingleton.elim _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.OutputLayout
