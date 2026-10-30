import bench.tritonbench_g.triton_conv2d_fwd.IO
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural
noncomputable section

def logicalConvolution (s : Shape) (xs : LogicalInputs s) : LogicalOutputs s :=
  fun g n y x f => .scalar
    (∑ c ∈ Finset.range s.cg,∑ h ∈ Finset.range s.kh,∑ w ∈ Finset.range s.kw,
      xs.paddedInput s n.val (g.val*s.cg+c) ((y.val : Int)*s.sh-s.ph+h)
        ((x.val : Int)*s.sw-s.pw+w) * xs.filter s (g.val*s.og+f.val) c h w)

theorem spatial_coordinates (s : Shape) (valid : Valid s)
    (n : Fin s.n) (h : Fin s.oh) (w : Fin s.ow) :
    (spatialIndex s n h w).val/(s.oh*s.ow)=n.val ∧
      (spatialIndex s n h w).val/s.ow%s.oh=h.val ∧
      (spatialIndex s n h w).val%s.ow=w.val := by
  have inner : w.val+s.ow*h.val < s.oh*s.ow := (finProdFinEquiv (h,w)).isLt
  change (w.val+s.ow*h.val+s.oh*s.ow*n.val)/(s.oh*s.ow)=n.val ∧
    (w.val+s.ow*h.val+s.oh*s.ow*n.val)/s.ow%s.oh=h.val ∧
    (w.val+s.ow*h.val+s.oh*s.ow*n.val)%s.ow=w.val
  constructor
  · rw [Nat.add_mul_div_left _ _ (Nat.mul_pos valid.oh valid.ow),Nat.div_eq_of_lt inner,Nat.zero_add]
  · have eq : w.val+s.ow*h.val+s.oh*s.ow*n.val = w.val+s.ow*(h.val+s.oh*n.val) := by ring
    rw [eq,Nat.add_mul_div_left _ _ valid.ow,Nat.div_eq_of_lt w.isLt,Nat.zero_add]
    constructor
    · simp only [Nat.add_mul_mod_self_left,Nat.mod_eq_of_lt h.isLt]
    · simp only [Nat.add_mul_mod_self_left,Nat.mod_eq_of_lt w.isLt]

theorem formula_eq (s : Shape) (valid : Valid s) :
    (logicalOutputs s) ∘ mathematical s ∘ (logicalInputs s).symm = logicalConvolution s := by
  funext xs g n y x f
  have coords := spatial_coordinates s valid n y x
  change KernelNumeric.Value.scalar (Conv2d.mathematical s
    (logicalScalars KernelReal.operations s ((logicalInputs s).symm xs))
    ⟨g.val,(spatialIndex s n y x).val,f.val⟩) = logicalConvolution s xs g n y x f
  simp only [Conv2d.mathematical,logicalConvolution,LogicalInputs.paddedInput,LogicalInputs.filter,
    coords.1,coords.2.1,coords.2.2]

theorem logical_iff (s : Shape) (valid : Valid s) (flags : Flags) :
    (logicalRealIO s flags).Implements (logicalConvolution s) ↔
      (realIO s flags).Implements (mathematical s) := by
  have eq : (logicalRealIO s flags).lift (logicalConvolution s) = mathematical s := by
    rw [← formula_eq s valid]
    funext xs
    change (logicalOutputs s).symm ((logicalOutputs s) (mathematical s
      ((logicalInputs s).symm ((logicalInputs s) xs)))) = mathematical s xs
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]
  change (realIO s flags).Implements _ ↔ _
  rw [eq]
end
end VeriTile.Bench.Repairs.Conv2d.ConvIO
