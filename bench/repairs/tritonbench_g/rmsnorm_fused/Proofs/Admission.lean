import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Contract
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Geometry
import Mathlib.Data.List.Basic

namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile.Torch
noncomputable section
set_option maxHeartbeats 3000000

private theorem representation_iff (x : TensorInfo) : Representation x ↔
    x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda⟩
  · rintro ⟨a,b,c⟩; exact ⟨a,b,c⟩
private theorem canonical_iff (x : TensorInfo) : Canonical x ↔
    x.isContiguous=Bool.true ∧ x.storageOffset=0 ∧ x.isNeg=Bool.false ∧ x.isConj=Bool.false := by
  constructor
  · intro h; exact ⟨h.contiguous,h.offset,h.negative,h.conjugate⟩
  · rintro ⟨a,b,c,d⟩; exact ⟨a,b,c,d⟩

theorem contract_iff (x weight : TensorInfo) (eps : EpsilonInfo) : InputContract x weight eps ↔
    Representation x ∧ Representation weight ∧ x.device=weight.device ∧
    1≤x.shape.length ∧ 0<width x ∧ weight.shape=[width x] ∧ Canonical x ∧ Canonical weight ∧
    eps.pythonFloat=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value ∧
    width x≤block (width x) ∧ x.numel≤(2^63-1)/4 ∧ rows x≤2^31-1 ∧
    (x.numel≠0 → x.dataPtr.toNat%4=0 ∧ weight.dataPtr.toNat%4=0) := by
  constructor
  · intro h
    exact ⟨h.input,h.weightRepresentation,h.device,h.rank,h.widthPositive,h.weightShape,h.inputCanonical,
      h.weightCanonical,h.epsilonType,h.epsilonFinite,h.epsilonPositive,h.block,h.byteExtent,h.grid,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o⟩
    exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o⟩

theorem checkTensorInput_iff (x weight : TensorInfo) (eps : EpsilonInfo) :
    checkTensorInput x weight eps=.ok () ↔ InputContract x weight eps := by
  classical
  rw [contract_iff,representation_iff,representation_iff,canonical_iff,canonical_iff]
  simp [checkTensorInput,TensorMetadata.isCuda,ite_eq_iff]
  have rank : (1≤x.shape.length) ↔ x.shape≠[] := by
    cases h : x.shape <;> simp
  simp only [rank,Nat.pos_iff_ne_zero]
  tauto

theorem checkInput_iff (x weight : Argument) (eps : EpsilonInfo) :
    checkInput x weight eps=.ok () ↔ Admitted x weight eps := by
  cases x <;> cases weight <;> simp [checkInput,Admitted,checkTensorInput_iff]

def geometry (x : TensorInfo) : Shape := ⟨x.shape.dropLast,width x,block (width x)⟩

theorem geometry_shape (x : TensorInfo) (hr : 1≤x.shape.length) : (geometry x).dims=x.shape := by
  have nonempty : x.shape≠[] := by intro h; simp [h] at hr
  simp only [geometry,Shape.dims,width,List.getLast?_eq_some_getLast nonempty,Option.getD_some]
  exact List.dropLast_append_getLast nonempty

theorem geometry_size (x : TensorInfo) (hr : 1≤x.shape.length) :
    (geometry x).rows*(geometry x).width=x.numel := by
  rw [←dimensions_product,geometry_shape x hr]
  rfl

theorem geometry_rows (x : TensorInfo) (hr : 1≤x.shape.length) (hp : 0<width x) :
    (geometry x).rows=rows x := by
  unfold rows
  rw [←geometry_size x hr]
  change (geometry x).rows=(geometry x).rows*width x/width x
  rw [Nat.mul_div_cancel _ hp]

theorem block_bound (n : Nat) : block n≤16384 := by
  unfold block
  by_cases h0 : n≤1
  · rw [if_pos h0]
    decide
  · rw [if_neg h0]
    by_cases h1 : n≤2
    · rw [if_pos h1]
      decide
    · rw [if_neg h1]
      by_cases h2 : n≤4
      · rw [if_pos h2]
        decide
      · rw [if_neg h2]
        by_cases h3 : n≤8
        · rw [if_pos h3]
          decide
        · rw [if_neg h3]
          by_cases h4 : n≤16
          · rw [if_pos h4]
            decide
          · rw [if_neg h4]
            by_cases h5 : n≤32
            · rw [if_pos h5]
              decide
            · rw [if_neg h5]
              by_cases h6 : n≤64
              · rw [if_pos h6]
                decide
              · rw [if_neg h6]
                by_cases h7 : n≤128
                · rw [if_pos h7]
                  decide
                · rw [if_neg h7]
                  by_cases h8 : n≤256
                  · rw [if_pos h8]
                    decide
                  · rw [if_neg h8]
                    by_cases h9 : n≤512
                    · rw [if_pos h9]
                      decide
                    · rw [if_neg h9]
                      by_cases h10 : n≤1024
                      · rw [if_pos h10]
                        decide
                      · rw [if_neg h10]
                        by_cases h11 : n≤2048
                        · rw [if_pos h11]
                          decide
                        · rw [if_neg h11]
                          by_cases h12 : n≤4096
                          · rw [if_pos h12]
                            decide
                          · rw [if_neg h12]
                            by_cases h13 : n≤8192
                            · rw [if_pos h13]
                              decide
                            · rw [if_neg h13]

theorem geometry_valid (x w : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps) :
    (geometry x).Valid :=
  ⟨c.widthPositive,c.block,block_bound _,by rw [geometry_size x c.rank]; exact c.byteExtent⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused
