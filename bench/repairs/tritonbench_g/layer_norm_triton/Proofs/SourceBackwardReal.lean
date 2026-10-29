import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardGroupReal
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardReal
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceBackwardStorage
noncomputable section
variable {x w dy r mu : Input}
set_option maxHeartbeats 2000000

theorem final_dw (model : Model ℝ) (laws : SourceFinalReal.Laws model)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ))
    (i : Fin (c.last.buffers.output 0).count) :
    finalValues (graphFinal model Memory.realAPI c memory) c 0 i=
      ∑ group ∈ Finset.range (min (Geometry.groups N) M),
        Memory.realAPI.read (memory (c.first.buffers.output 1).owner) ((c.first.buffers.output 1).address (group*N+i.val)) := by
  let s := initialFinal Memory.realAPI c memory (i.val/128)
  let lane : TileIndex [128] := (⟨i.val%128,Nat.mod_lt _ (by decide)⟩,PUnit.unit)
  have hc : SourceFinalGraph.columns s 128 lane=i.val := by
    simp [SourceFinalGraph.columns,s,initialFinal,lane,Nat.mul_comm,Nat.div_add_mod]
  have hi : SourceFinalGraph.columns s 128 lane<N := by rw [hc]; simpa only [c.last.dwCount] using i.isLt
  have hv := SourceFinalResult.stores_dw model s N 32 128
    (SourceFinalGraph.fold model s (min (Geometry.groups N) M) N 32 128 0 (SourceFinalGraph.zero model 32 128)) lane hi
  have hr := (SourceFinalReal.reduced_values model laws s (min (Geometry.groups N) M) N 128 lane hi).1
  rw [hc] at hv hr
  have hload (group : Nat) : (s.mem "DW" (group*N+i.val)).read .real=
      Memory.realAPI.read (memory (c.first.buffers.output 1).owner) ((c.first.buffers.output 1).address (group*N+i.val)) := by
    simp [s,initialFinal]
  simp only [hload] at hr
  have ho := congrArg (fun cell : Cell ℝ => cell.read .real) hv
  simp only [Cell.read_same] at ho
  rw [hr] at ho
  exact ho

theorem final_db (model : Model ℝ) (laws : SourceFinalReal.Laws model)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ))
    (i : Fin (c.last.buffers.output 1).count) :
    finalValues (graphFinal model Memory.realAPI c memory) c 1 i=
      ∑ group ∈ Finset.range (min (Geometry.groups N) M),
        Memory.realAPI.read (memory (c.first.buffers.output 2).owner) ((c.first.buffers.output 2).address (group*N+i.val)) := by
  let s := initialFinal Memory.realAPI c memory (i.val/128)
  let lane : TileIndex [128] := (⟨i.val%128,Nat.mod_lt _ (by decide)⟩,PUnit.unit)
  have hc : SourceFinalGraph.columns s 128 lane=i.val := by
    simp [SourceFinalGraph.columns,s,initialFinal,lane,Nat.mul_comm,Nat.div_add_mod]
  have hi : SourceFinalGraph.columns s 128 lane<N := by rw [hc]; simpa only [c.last.dbCount] using i.isLt
  have hv := SourceFinalResult.stores_db model s N 32 128
    (SourceFinalGraph.fold model s (min (Geometry.groups N) M) N 32 128 0 (SourceFinalGraph.zero model 32 128)) lane hi
  have hr := (SourceFinalReal.reduced_values model laws s (min (Geometry.groups N) M) N 128 lane hi).2
  rw [hc] at hv hr
  have hload (group : Nat) : (s.mem "DB" (group*N+i.val)).read .real=
      Memory.realAPI.read (memory (c.first.buffers.output 2).owner) ((c.first.buffers.output 2).address (group*N+i.val)) := by
    simp [s,initialFinal]
  simp only [hload] at hr
  have ho := congrArg (fun cell : Cell ℝ => cell.read .real) hv
  simp only [Cell.read_same] at ho
  rw [hr] at ho
  exact ho

theorem scratch_dw (model : Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (hb : N≤B)
    (group col : Nat) (hg : group<min (Geometry.groups N) M) (hc : col<N) :
    Memory.realAPI.read ((afterGroup model B Memory.realAPI Neg.neg c memory) (c.first.buffers.output 1).owner)
      ((c.first.buffers.output 1).address (group*N+col))=
      ((groupRows M (Geometry.groups N) group).map (SourceBackwardGroupReal.weightTerm c memory col)).sum := by
  have hn : 0<N := by omega
  have hi := (Geometry.partial_offset_safe (c.first.buffers.output 1) M N (Geometry.groups N) group col c.first.dwCount hg hc).1
  have hv := KernelMemory.run_read c.first.buffers Memory.realAPI Neg.neg
    (groupValues (graphGroup model B Memory.realAPI Neg.neg c memory) c) memory 1 ⟨group*N+col,hi⟩
  rw [dense_logical Memory.realAPI Neg.neg _ _ _ hi,SourceBackwardGroupReal.values_real model laws B c memory hn hb] at hv
  have hd : (group*N+col)/N=group := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt hc,Nat.zero_add]
  have hm : (group*N+col)%N=col := by simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
  simpa only [afterGroup,SourceBackwardGroupReal.values_real model laws B c memory hn hb,
    SourceBackwardGroupReal.values,hd,hm] using hv

theorem scratch_db (model : Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (hb : N≤B)
    (group col : Nat) (hg : group<min (Geometry.groups N) M) (hc : col<N) :
    Memory.realAPI.read ((afterGroup model B Memory.realAPI Neg.neg c memory) (c.first.buffers.output 2).owner)
      ((c.first.buffers.output 2).address (group*N+col))=
      ((groupRows M (Geometry.groups N) group).map (SourceBackwardGroupReal.biasTerm c memory col)).sum := by
  have hn : 0<N := by omega
  have hi := (Geometry.partial_offset_safe (c.first.buffers.output 2) M N (Geometry.groups N) group col c.first.dbCount hg hc).1
  have hv := KernelMemory.run_read c.first.buffers Memory.realAPI Neg.neg
    (groupValues (graphGroup model B Memory.realAPI Neg.neg c memory) c) memory 2 ⟨group*N+col,hi⟩
  rw [dense_logical Memory.realAPI Neg.neg _ _ _ hi,SourceBackwardGroupReal.values_real model laws B c memory hn hb] at hv
  have hd : (group*N+col)/N=group := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt hc,Nat.zero_add]
  have hm : (group*N+col)%N=col := by simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
  simpa only [afterGroup,SourceBackwardGroupReal.values_real model laws B c memory hn hb,
    SourceBackwardGroupReal.values,hd,hm] using hv

theorem values_real (model : Model ℝ) (laws : SourceFinalReal.Laws model) (B : Nat)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (hn : 0<N) (hb : N≤B) :
    SourceBackwardResult.values model B Memory.realAPI Neg.neg c memory=KernelBackward.realValues c memory := by
  funext k i
  fin_cases k
  · exact congrArg (fun v => v 0 i) (SourceBackwardGroupReal.values_real model laws.toLaws B c memory hn hb)
  · have hc : i.val<N := by simpa only [KernelBackward.outputs,c.last.dwCount] using i.isLt
    change finalValues _ c 0 i=_
    rw [final_dw model laws]
    calc
      _ = ∑ group ∈ Finset.range (min (Geometry.groups N) M),
          ((groupRows M (Geometry.groups N) group).map (SourceBackwardGroupReal.weightTerm c memory i.val)).sum := by
            apply Finset.sum_congr rfl
            intro group hg
            exact scratch_dw model laws.toLaws B c memory hb group i.val (Finset.mem_range.mp hg) hc
      _ = ∑ row ∈ Finset.range M,SourceBackwardGroupReal.weightTerm c memory i.val row :=
            RealProof.grouped_sum M (Geometry.groups N) (Geometry.groups_bounds N).1 _
      _ = _ := by simp [KernelBackward.realValues,RealProof.Spec.weightGradient,
        VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec,hc,
        SourceBackwardGroupReal.weightTerm,KernelBackward.initialStats]
  · have hc : i.val<N := by simpa only [KernelBackward.outputs,c.last.dbCount] using i.isLt
    change finalValues _ c 1 i=_
    rw [final_db model laws]
    calc
      _ = ∑ group ∈ Finset.range (min (Geometry.groups N) M),
          ((groupRows M (Geometry.groups N) group).map (SourceBackwardGroupReal.biasTerm c memory i.val)).sum := by
            apply Finset.sum_congr rfl
            intro group hg
            exact scratch_db model laws.toLaws B c memory hb group i.val (Finset.mem_range.mp hg) hc
      _ = ∑ row ∈ Finset.range M,SourceBackwardGroupReal.biasTerm c memory i.val row :=
            RealProof.grouped_sum M (Geometry.groups N) (Geometry.groups_bounds N).1 _
      _ = _ := by simp [KernelBackward.realValues,RealProof.Spec.biasGradient,
        VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec,hc,SourceBackwardGroupReal.biasTerm]

#axiomsClean values_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardReal
