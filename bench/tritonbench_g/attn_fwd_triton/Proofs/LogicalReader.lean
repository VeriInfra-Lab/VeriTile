import bench.tritonbench_g.attn_fwd_triton.Proofs.RealImplementation
import bench.tritonbench_g.attn_fwd_triton.IO

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
open VeriTile Triton AttnFwdTriton Verified
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem projection_iff (A : FlatAlloc) (p : Parameters) :
    (attentionIO A p ⊨ fun xs c => RealProofs.expected p.heads p.context p.headDim
      c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val xs.tensors) ↔
    (programIO A p).Implements (formula p) := by
  change (programIO A p).Implements
    (fun vals _ j => RealProofs.expected p.heads p.context p.headDim
      ((coordinates p).symm j).1.val ((coordinates p).symm j).2.1.val
      ((coordinates p).symm j).2.2.1.val ((coordinates p).symm j).2.2.2.val
      (logicalInputs p ((logicalInputsEquiv p).symm (logicalInputsEquiv p vals)))) ↔ _
  simp only [Equiv.symm_apply_apply]
  rfl

theorem scaleAt_packed (values : Fin batches → Fin heads → Fin blocks → ℝ)
    (batch head block : Nat) (hz : batch < batches) (hh : head < heads) (hb : block < blocks) :
    scaleAt values batch head block =
      scale (fun i : Fin (batches*heads*blocks) =>
        let c := (scaleIndex batches heads blocks).symm i
        values c.1 c.2.1 c.2.2) ((batch*heads+head)*blocks+block) := by
  have bound : (batch*heads+head)*blocks+block < batches*heads*blocks := by
    simpa [scaleIndex,finProdFinEquiv,Nat.add_comm,Nat.mul_comm] using
      (scaleIndex batches heads blocks (⟨batch,hz⟩,⟨head,hh⟩,⟨block,hb⟩)).isLt
  have idx : scaleIndex batches heads blocks (⟨batch,hz⟩,⟨head,hh⟩,⟨block,hb⟩) =
      ⟨(batch*heads+head)*blocks+block,bound⟩ := by
    apply Fin.ext
    simp [scaleIndex,finProdFinEquiv,Nat.add_comm,Nat.mul_comm]
  simp only [scaleAt,hz,hh,hb,and_self,dif_pos,scale,bound]
  rw [←idx,Equiv.symm_apply_apply]

def blockScore (p : Parameters) (xs : LogicalInputs p) (batch head row key : Nat) : ℝ :=
  ((List.range p.headDim).map (fun d =>
    xs.tensors.query batch head row d * xs.tensors.key batch head key d)).sum *
    xs.queryScaleAt batch head (row/128) * xs.keyScaleAt batch head (key/64)

def causalFormula (p : Parameters) (xs : LogicalInputs p) (c : Coordinate p) : ℝ :=
  let pairs := (List.range (c.2.2.1.val+1)).map (fun n =>
    (blockScore p xs c.1.val c.2.1.val c.2.2.1.val n,
      xs.tensors.value c.1.val c.2.1.val n c.2.2.2.val))
  (pairs.map (fun sv => Real.exp (Real.log 2 * sv.1) * sv.2)).sum /
    (pairs.map (fun sv => Real.exp (Real.log 2 * sv.1))).sum

theorem blockScore_eq (p : Parameters) (xs : LogicalInputs p) (batch head row key : Nat)
    (hz : batch < p.batches) (hh : head < p.heads) (hr : row < p.context) (hk : key < p.context) :
    blockScore p xs batch head row key =
      RealProofs.score p.heads p.context p.headDim batch head row key xs.tensors := by
  unfold blockScore RealProofs.score LogicalInputs.queryScaleAt LogicalInputs.keyScaleAt
  rw [scaleAt_packed xs.queryScale batch head (row/128) hz hh (by omega),
    scaleAt_packed xs.keyScale batch head (key/64) hz hh (by omega)]
  rfl

theorem causalFormula_eq (p : Parameters) : causalFormula p =
    (fun xs c => RealProofs.expected p.heads p.context p.headDim
      c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val xs.tensors) := by
  funext xs c
  have pairs_eq : (List.range (c.2.2.1.val+1)).map (fun n =>
      (blockScore p xs c.1.val c.2.1.val c.2.2.1.val n,
        xs.tensors.value c.1.val c.2.1.val n c.2.2.2.val)) =
      RealProofs.pairs p.heads p.context p.headDim c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val xs.tensors := by
    apply List.map_congr_left
    intro n hn
    have hn := List.mem_range.mp hn
    rw [blockScore_eq p xs _ _ _ _ c.1.isLt c.2.1.isLt c.2.2.1.isLt (by omega)]
  unfold causalFormula
  rw [pairs_eq]
  rfl

theorem logical_iff (A : FlatAlloc) (p : Parameters) :
    (attentionIO A p ⊨ causalFormula p) ↔ (programIO A p).Implements (formula p) := by
  have eq := congrArg (attentionIO A p).Implements (causalFormula_eq p)
  exact eq.to_iff.trans (projection_iff A p)

theorem logical_implements (A : FlatAlloc) (p : Parameters) (stage : p.stage=3)
    (nonoverlap : OutputLayout.NonOverlapping p) :
    attentionIO A p ⊨ causalFormula p :=
  (logical_iff A p).mpr (implements A p stage nonoverlap)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
