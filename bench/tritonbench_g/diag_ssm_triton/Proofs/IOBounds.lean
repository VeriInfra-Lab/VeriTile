import bench.tritonbench_g.diag_ssm_triton.Proofs.IOBindings
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_BackwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexForwardSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardSafety

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem forward_input_bound (complex : Bool) (p : ForwardLayout) (length : Nat)
    (u : BlockState) (v : (forwardIO complex p length).Ctx) (bounds : RegionBounds)
    (hb : (forwardIO complex p length).RegionalBounds bounds u v)
    (role : Fin 3) (c : Fin (components complex)) (k : Fin (forwardArity p.B length role))
    (ha : active p.batch p.dim p.B (u.pids 0) (forwardLane p.B role k.val)) :
    forwardAddress p (u.pids 0) role k.val*components complex+c.val < bounds (forwardBuffer p role) := by
  let kk : Fin ((forwardIO complex p length).iarity (pair role c)) :=
    ⟨k.val,by simpa only [forwardIO,divNat_pair] using k.isLt⟩
  have hm : (forwardIO complex p length).imask (pair role c) v (u.pids 0) (u.pids 1) (u.pids 2) kk := by
    simpa only [forwardIO,divNat_pair] using ha
  have h := hb.1 (pair role c) kk hm
  simpa only [forwardIO,divNat_pair,modNat_pair] using h

theorem forward_output_bound (complex : Bool) (p : ForwardLayout) (length : Nat)
    (u : BlockState) (v : (forwardIO complex p length).Ctx) (bounds : RegionBounds)
    (hb : (forwardIO complex p length).RegionalBounds bounds u v)
    (c : Fin (components complex)) (k : Fin (length*p.B))
    (ha : active p.batch p.dim p.B (u.pids 0) (k.val%p.B)) :
    streamAddress p.batch p.dim p.B (u.pids 0) k.val*components complex+c.val < bounds p.y := hb.2.1 c k ha

theorem backward_input_bound (complex : Bool) (p : BackwardLayout) (length : Nat)
    (u : BlockState) (v : (backwardIO complex p length).Ctx) (bounds : RegionBounds)
    (hb : (backwardIO complex p length).RegionalBounds bounds u v)
    (role : Fin 4) (c : Fin (components complex)) (k : Fin (backwardArity p.B length role))
    (hz : role.val=0 → 0<length)
    (ha : active p.batch p.dim p.B (u.pids 0) (backwardLane p.B role k.val)) :
    backwardAddress p (u.pids 0) role k.val*components complex+c.val < bounds (backwardBuffer p role) := by
  let kk : Fin ((backwardIO complex p length).iarity (pair role c)) :=
    ⟨k.val,by simpa only [backwardIO,divNat_pair] using k.isLt⟩
  have hm : (backwardIO complex p length).imask (pair role c) v (u.pids 0) (u.pids 1) (u.pids 2) kk := by
    simpa only [backwardIO,divNat_pair] using And.intro hz ha
  have h := hb.1 (pair role c) kk hm
  simpa only [backwardIO,divNat_pair,modNat_pair] using h

theorem backward_output_bound (complex : Bool) (p : BackwardLayout) (length : Nat)
    (u : BlockState) (v : (backwardIO complex p length).Ctx) (bounds : RegionBounds)
    (hb : (backwardIO complex p length).RegionalBounds bounds u v)
    (role : Fin 3) (c : Fin (components complex)) (k : Fin (backwardOutArity p.B length role))
    (ha : active p.batch p.dim p.B (u.pids 0) (backwardOutLane p.B role k.val)) :
    backwardOutAddress p (u.pids 0) role k.val*components complex+c.val < bounds (backwardOutBuffer p role) := by
  let kk : Fin ((backwardIO complex p length).oarity (pair role c)) :=
    ⟨k.val,by simpa only [backwardIO,divNat_pair] using k.isLt⟩
  have hm : (backwardIO complex p length).omask (pair role c) v (u.pids 0) (u.pids 1) (u.pids 2) kk := by
    simpa only [backwardIO,divNat_pair] using ha
  have h := hb.2.1 (pair role c) kk hm
  simpa only [backwardIO,divNat_pair,modNat_pair] using h
 theorem forward_bounds_real (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.false p length).Ctx) (bounds : RegionBounds)
    (hb : (forwardIO Bool.false p length).RegionalBounds bounds u v) :
    ForwardIO.Bounds p u length bounds := by
  constructor
  · intro j hj
    have h := forward_input_bound Bool.false p length u v bounds hb 0 (0 : Fin 1) (by exact j)
      (by simpa [forwardLane, active, column, ForwardIO.active, ForwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ForwardIO.offset, ForwardIO.column] using h
  · intro j hj
    have h := forward_input_bound Bool.false p length u v bounds hb 1 (0 : Fin 1) (by exact j)
      (by simpa [forwardLane, active, column, ForwardIO.active, ForwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ForwardIO.offset, ForwardIO.column] using h
  · intro t ht j hj
    have h := forward_input_bound Bool.false p length u v bounds hb 2 (0 : Fin 1) (by exact pair (⟨t,ht⟩ : Fin length) j)
      (by simpa [forwardLane, active, column, ForwardIO.active, ForwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ForwardIO.offset, ForwardIO.column] using h
  · intro t ht j hj
    have h := forward_output_bound Bool.false p length u v bounds hb (0 : Fin 1) (pair (⟨t,ht⟩ : Fin length) j)
      (by simpa [forwardLane, active, column, ForwardIO.active, ForwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ForwardIO.offset, ForwardIO.column] using h

 theorem forward_bounds_complex (p : ForwardLayout) (length : Nat) (u : BlockState)
    (v : (forwardIO Bool.true p length).Ctx) (bounds : RegionBounds)
    (hb : (forwardIO Bool.true p length).RegionalBounds bounds u v) :
    ComplexForwardSafety.Bounds p u length bounds := by
  constructor
  · intro j hj
    have h := forward_input_bound Bool.true p length u v bounds hb 0 (1 : Fin 2) (by exact j.1)
      (by simpa [forwardLane, active, column, ComplexForwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexForwardLoop.offset, ComplexForwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro j hj
    have h := forward_input_bound Bool.true p length u v bounds hb 1 (1 : Fin 2) (by exact j.1)
      (by simpa [forwardLane, active, column, ComplexForwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexForwardLoop.offset, ComplexForwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro t ht j hj
    have h := forward_input_bound Bool.true p length u v bounds hb 2 (1 : Fin 2) (by exact pair (⟨t,ht⟩ : Fin length) j.1)
      (by simpa [forwardLane, active, column, ComplexForwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexForwardLoop.offset, ComplexForwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro t ht j hj
    have h := forward_output_bound Bool.true p length u v bounds hb (1 : Fin 2) (pair (⟨t,ht⟩ : Fin length) j.1)
      (by simpa [forwardLane, active, column, ComplexForwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [forwardAddress, forwardBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexForwardLoop.offset, ComplexForwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h

 theorem backward_bounds_real (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.false p length).Ctx) (bounds : RegionBounds)
    (hb : (backwardIO Bool.false p length).RegionalBounds bounds u v) :
    BackwardIO.Bounds p u length bounds := by
  constructor
  · intro hp j hj
    have h := backward_input_bound Bool.false p length u v bounds hb 0 (0 : Fin 1) (by exact j) (by omega)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro j hj
    have h := backward_input_bound Bool.false p length u v bounds hb 1 (0 : Fin 1) (by exact j) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro t ht j hj
    have h := backward_input_bound Bool.false p length u v bounds hb 2 (0 : Fin 1) (by exact pair (⟨t,by omega⟩ : Fin (length-1)) j) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro t ht j hj
    have h := backward_input_bound Bool.false p length u v bounds hb 3 (0 : Fin 1) (by exact pair (⟨t,ht⟩ : Fin length) j) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro t ht j hj
    have h := backward_output_bound Bool.false p length u v bounds hb 0 (0 : Fin 1) (by exact pair (⟨t,ht⟩ : Fin length) j)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro j hj
    have h := backward_output_bound Bool.false p length u v bounds hb 1 (0 : Fin 1) (by exact j)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h
  · intro j hj
    have h := backward_output_bound Bool.false p length u v bounds hb 2 (0 : Fin 1) (by exact j)
      (by simpa [backwardLane, backwardOutLane, active, column, BackwardIO.active, BackwardIO.column, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, BackwardIO.offset, BackwardIO.column] using h

 theorem backward_bounds_complex (p : BackwardLayout) (length : Nat) (u : BlockState)
    (v : (backwardIO Bool.true p length).Ctx) (bounds : RegionBounds)
    (hb : (backwardIO Bool.true p length).RegionalBounds bounds u v) :
    ComplexBackwardSafety.Bounds p u length bounds := by
  constructor
  · intro hp j hj
    have h := backward_input_bound Bool.true p length u v bounds hb 0 (1 : Fin 2) (by exact j.1) (by omega)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro j hj
    have h := backward_input_bound Bool.true p length u v bounds hb 1 (1 : Fin 2) (by exact j.1) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro t ht j hj
    have h := backward_input_bound Bool.true p length u v bounds hb 2 (1 : Fin 2) (by exact pair (⟨t,by omega⟩ : Fin (length-1)) j.1) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro t ht j hj
    have h := backward_input_bound Bool.true p length u v bounds hb 3 (1 : Fin 2) (by exact pair (⟨t,ht⟩ : Fin length) j.1) (by simp)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro t ht j hj
    have h := backward_output_bound Bool.true p length u v bounds hb 0 (1 : Fin 2) (by exact pair (⟨t,ht⟩ : Fin length) j.1)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro j hj
    have h := backward_output_bound Bool.true p length u v bounds hb 1 (1 : Fin 2) (by exact j.1)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h
  · intro j hj
    have h := backward_output_bound Bool.true p length u v bounds hb 2 (1 : Fin 2) (by exact j.1)
      (by simpa [backwardLane, backwardOutLane, active, column, ComplexBackwardLoop.mask, ComplexForwardInit.active, ComplexForwardInit.columns, decide_eq_true_eq, pair_mod] using hj)
    try dsimp only [forwardAddress,forwardBuffer,backwardAddress,backwardBuffer,backwardOutAddress,backwardOutBuffer] at h
    try simp only [streamAddress] at h
    try simp only [pair_div,pair_mod] at h
    simpa [backwardAddress, backwardBuffer, backwardOutAddress, backwardOutBuffer, components, column, streamAddress, pair_div, pair_mod, Nat.mul_one, Nat.add_zero, ComplexBackwardLoop.offset, ComplexBackwardLoop.columns, ComplexForwardStep.offset, ComplexForwardInit.columns] using h

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
