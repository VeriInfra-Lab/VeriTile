import bench.tritonbench_g.fused_rotary_embedding.Proofs.PublicIO
import bench.tritonbench_g.fused_rotary_embedding.IO

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

/-- Full old/new domain and judgment equivalence for any independent target. -/
theorem logical_iff (rest : Algebra ℝ) (c : Config)
    (target : ChannelValues ℝ (inputChannels c.shape) → ChannelValues ℝ (outputChannels c.shape)) :
    (logicalIO rest c).ImplementsOn (logicalMetadataDomain c)
      (fun xs => (outputCoordinates ℝ c.shape) (target ((inputCoordinates ℝ c.shape).symm xs))) ↔
    (io rest realOps id (fun _ => id) c).ImplementsOn (metadataDomain c) target := by
  have eq : (logicalIO rest c).lift
      (fun xs => (outputCoordinates ℝ c.shape) (target ((inputCoordinates ℝ c.shape).symm xs))) = target := by
    funext xs
    change (outputCoordinates ℝ c.shape).symm ((outputCoordinates ℝ c.shape)
      (target ((inputCoordinates ℝ c.shape).symm ((inputCoordinates ℝ c.shape) xs)))) = target xs
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]
  change (io rest realOps id (fun _ => id) c).ImplementsOn
    (fun xs => metadataDomain c ((inputCoordinates ℝ c.shape).symm ((inputCoordinates ℝ c.shape) xs))) _ ↔ _
  simp only [Equiv.symm_apply_apply,eq]

/-- The tensor row accessor agrees with every logical tensor cell. Its total
extension cannot affect valid split-half formulas. -/
theorem flat_tensor_read (l m n : Nat) (xs : Fin l → Fin m → Fin n → ℝ)
    (t : Fin l) (h : Fin m) (d : Fin n) :
    vectorAt ((flatTensor ℝ .real l m n).symm xs) ((t.val*m+h.val)*n+d.val) = xs t h d := by
  let index : Fin (l*m*n) := finProdFinEquiv (finProdFinEquiv (t,h),d)
  have value : index.val = (t.val*m+h.val)*n+d.val := by
    dsimp [index]
    ring
  have bound : (t.val*m+h.val)*n+d.val<l*m*n := value ▸ index.isLt
  unfold vectorAt
  rw [dif_pos bound]
  have he : (⟨(t.val*m+h.val)*n+d.val,bound⟩ : Fin (l*m*n)) = index := Fin.ext value.symm
  rw [he]
  change (let p := finProdFinEquiv.symm (finProdFinEquiv (finProdFinEquiv (t,h),d));
    let q := finProdFinEquiv.symm p.1; xs q.1 q.2 p.2) = xs t h d
  simp only [Equiv.symm_apply_apply]

theorem tensorInput_q (s : Shape) (xs : RotaryInputs ℝ s)
    (t : Fin s.tokens) (h : Fin s.headsQ) (d : Fin s.width) :
    tensorInput s xs .q t.val h.val d.val = xs.q t h d := by
  change vectorAt ((flatTensor ℝ .real s.tokens s.headsQ s.width).symm xs.q)
    ((t.val*s.headsQ+h.val)*s.width+d.val) = _
  exact flat_tensor_read _ _ _ _ t h d

theorem tensorInput_k (s : Shape) (xs : RotaryInputs ℝ s)
    (t : Fin s.tokens) (h : Fin s.headsK) (d : Fin s.width) :
    tensorInput s xs .k t.val h.val d.val = xs.k t h d := by
  change vectorAt ((flatTensor ℝ .real s.tokens s.headsK s.width).symm xs.k)
    ((t.val*s.headsK+h.val)*s.width+d.val) = _
  exact flat_tensor_read _ _ _ _ t h d

theorem tensorInput_v (s : Shape) (xs : RotaryInputs ℝ s)
    (t : Fin s.tokens) (h : Fin s.headsK) (d : Fin s.width) :
    tensorInput s xs .v t.val h.val d.val = xs.v t h d := by
  change vectorAt ((flatTensor ℝ .real s.tokens s.headsK s.width).symm xs.v)
    ((t.val*s.headsK+h.val)*s.width+d.val) = _
  exact flat_tensor_read _ _ _ _ t h d

 theorem flat_matrix_read (m n : Nat) (xs : Fin m → Fin n → ℝ) (t : Fin m) (d : Fin n) :
    vectorAt ((flatMatrix ℝ .real m n).symm xs) (t.val*n+d.val) = xs t d := by
  let index : Fin (m*n) := finProdFinEquiv (t,d)
  have value : index.val = t.val*n+d.val := by dsimp [index]; ring
  have bound : t.val*n+d.val<m*n := value ▸ index.isLt
  unfold vectorAt
  rw [dif_pos bound]
  have he : (⟨t.val*n+d.val,bound⟩ : Fin (m*n)) = index := Fin.ext value.symm
  rw [he]
  change (let i := finProdFinEquiv.symm (finProdFinEquiv (t,d)); xs i.1 i.2) = xs t d
  rw [Equiv.symm_apply_apply]

theorem tensorCos_at (s : Shape) (xs : RotaryInputs ℝ s)
    (t : Fin s.tokens) (d : Fin s.cosWidth) : tensorCos s xs t.val d.val = xs.cos t d := by
  change vectorAt ((flatMatrix ℝ .real s.tokens s.cosWidth).symm xs.cos) (t.val*s.cosWidth+d.val) = _
  exact flat_matrix_read _ _ _ t d

theorem tensorSin_at (s : Shape) (xs : RotaryInputs ℝ s)
    (t : Fin s.tokens) (d : Fin s.sinWidth) : tensorSin s xs t.val d.val = xs.sin t d := by
  change vectorAt ((flatMatrix ℝ .real s.tokens s.sinWidth).symm xs.sin) (t.val*s.sinWidth+d.val) = _
  exact flat_matrix_read _ _ _ t d

theorem logical_real (rest : Algebra ℝ) (c : Config) (valid : c.Valid) :
    (logicalIO rest c).ImplementsOn (logicalMetadataDomain c)
      (fun xs => (outputCoordinates ℝ c.shape)
        (outputFormula c.shape (formula c.shape (inputMemory c.shape ((inputCoordinates ℝ c.shape).symm xs))))) :=
  (logical_iff rest c _).mpr (implements rest realOps id (fun _ => id) c valid
    (fun xs => formula c.shape (inputMemory c.shape xs))
    (fun xs w hw high => pair_real_formula c.shape valid.shape w hw _ high))
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
