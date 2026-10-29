import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedDX
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedSums

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardGroupReal
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceBackwardStorage
noncomputable section
variable {x w dy r mu : Input}

def weightTerm (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (col row : Nat) : ℝ :=
  KernelForward.row Memory.realAPI Neg.neg dy memory N row col*
    ((KernelForward.row Memory.realAPI Neg.neg x memory N row col-inputValue Memory.realAPI Neg.neg mu memory row)*
      inputValue Memory.realAPI Neg.neg r memory row)

def biasTerm (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (col row : Nat) : ℝ :=
  KernelForward.row Memory.realAPI Neg.neg dy memory N row col

def values (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) :
    ∀ k : Fin 3,Fin (c.first.buffers.output k).count → ℝ
  | ⟨0,_⟩,i => KernelBackward.realValues c memory 0 i
  | ⟨1,_⟩,i => ((groupRows M (Geometry.groups N) (i.val/N)).map (weightTerm c memory (i.val%N))).sum
  | ⟨2,_⟩,i => ((groupRows M (Geometry.groups N) (i.val/N)).map (biasTerm c memory (i.val%N))).sum

theorem terms_initial (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (group col : Nat) :
    SourceGroupedSums.weightTerm (initialGroup Memory.realAPI Neg.neg c memory group) N col=weightTerm c memory col ∧
    SourceGroupedSums.biasTerm (initialGroup Memory.realAPI Neg.neg c memory group) N col=biasTerm c memory col := by
  constructor <;> funext row <;>
    simp [SourceGroupedSums.weightTerm,SourceGroupedSums.biasTerm,SourceGroupedReal.normalized,
      SourceGroupedReal.row,SourceGroupedReal.savedMean,SourceGroupedReal.savedInverse,
      initialGroup,weightTerm,biasTerm,KernelForward.row]

theorem dx_initial (model : Model ℝ) (laws : SourceForwardReal.Laws model)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ))
    (B group row col : Nat) (hb : N≤B) (hc : col<N) :
    SourceGroupedGraph.dx model (initialGroup Memory.realAPI Neg.neg c memory group) N N B row col=
      RealProof.Spec.gradient N (KernelForward.row Memory.realAPI Neg.neg x memory N row)
        (KernelForward.row Memory.realAPI Neg.neg dy memory N row) (inputValue Memory.realAPI Neg.neg w memory)
        (inputValue Memory.realAPI Neg.neg mu memory row) (inputValue Memory.realAPI Neg.neg r memory row) col := by
  rw [SourceGroupedReal.dx_real model laws _ N N B row col hb hc]
  congr 1 <;>
    simp [SourceGroupedReal.row,SourceGroupedReal.weight,SourceGroupedReal.savedMean,SourceGroupedReal.savedInverse,
      initialGroup,KernelForward.row]

theorem values_real (model : Model ℝ) (laws : SourceForwardReal.Laws model) (B : Nat)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World (Nat → ℝ)) (hn : 0<N) (hb : N≤B) :
    groupValues (graphGroup model B Memory.realAPI Neg.neg c memory) c=values c memory := by
  have hg := (Geometry.groups_bounds N).1
  have index (i : Nat) : i/N*N+i%N=i := by simpa [Nat.mul_comm] using Nat.div_add_mod i N
  funext k i
  fin_cases k
  · have himn : i.val<M*N := by rw [←c.first.dxCount]; exact i.isLt
    have hr : i.val/N<M := (Nat.div_lt_iff_lt_mul hn).mpr himn
    have hgroup : (i.val/N)%Geometry.groups N<min (Geometry.groups N) M := by
      have hm := Nat.mod_lt (i.val/N) hg
      have hl := Nat.mod_le (i.val/N) (Geometry.groups N)
      omega
    have hmem : i.val/N∈groupRows M (Geometry.groups N) ((i.val/N)%Geometry.groups N) :=
      (Geometry.group_residue M (Geometry.groups N) _ _ hg hgroup).mpr ⟨hr,rfl⟩
    have hv := SourceGroupedDX.final_dx model M N (Geometry.groups N) B (i.val/N) (i.val%N)
      (initialGroup Memory.realAPI Neg.neg c memory ((i.val/N)%Geometry.groups N)) hg hb (Nat.mod_lt _ hn)
      (by simpa [initialGroup] using hmem)
    simp only [groupValues,graphGroup,values]
    have ho := congrArg (fun cell : Cell ℝ => cell.read .real) hv
    simp only [Cell.read_same] at ho
    rw [dx_initial model laws c memory B _ _ _ hb (Nat.mod_lt _ hn)] at ho
    simpa only [index,KernelBackward.realValues,KernelBackward.initialStats] using ho
  · let lane : TileIndex [B] := (⟨i.val%N,lt_of_lt_of_le (Nat.mod_lt _ hn) hb⟩,PUnit.unit)
    have hv := SourceGroupedResult.stores_dw
      (SourceGroupedExecution.fold model N M N (Geometry.groups N) B (i.val/N) (i.val/N)
        (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) (SourceGroupedGraph.zero model B)).1
      N B (i.val/N)
      (SourceGroupedExecution.fold model N M N (Geometry.groups N) B (i.val/N) (i.val/N)
        (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) (SourceGroupedGraph.zero model B)).2
      lane (Nat.mod_lt _ hn)
    have hs := (SourceGroupedSums.fold_group model laws N M N (Geometry.groups N) B (i.val/N)
      (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) hg lane (Nat.mod_lt _ hn)).1
    rw [(terms_initial c memory (i.val/N) (i.val%N)).1] at hs
    have ho := congrArg (fun cell : Cell ℝ => cell.read .real) hv
    simp only [Cell.read_same] at ho
    rw [hs] at ho
    simpa only [groupValues,graphGroup,SourceGroupedExecution.final,initialGroup,ite_true,lane,index,values] using ho
  · let lane : TileIndex [B] := (⟨i.val%N,lt_of_lt_of_le (Nat.mod_lt _ hn) hb⟩,PUnit.unit)
    have hv := SourceGroupedResult.stores_db
      (SourceGroupedExecution.fold model N M N (Geometry.groups N) B (i.val/N) (i.val/N)
        (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) (SourceGroupedGraph.zero model B)).1
      N B (i.val/N)
      (SourceGroupedExecution.fold model N M N (Geometry.groups N) B (i.val/N) (i.val/N)
        (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) (SourceGroupedGraph.zero model B)).2
      lane (Nat.mod_lt _ hn)
    have hs := (SourceGroupedSums.fold_group model laws N M N (Geometry.groups N) B (i.val/N)
      (initialGroup Memory.realAPI Neg.neg c memory (i.val/N)) hg lane (Nat.mod_lt _ hn)).2
    rw [(terms_initial c memory (i.val/N) (i.val%N)).2] at hs
    have ho := congrArg (fun cell : Cell ℝ => cell.read .real) hv
    simp only [Cell.read_same] at ho
    rw [hs] at ho
    simpa only [groupValues,graphGroup,SourceGroupedExecution.final,initialGroup,ite_true,lane,index,values] using ho

#axiomsClean values_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardGroupReal
