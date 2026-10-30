import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicLogicalModel
import Mathlib.Logic.Equiv.Fin.Basic

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
set_option maxHeartbeats 2000000

/-- A proof-only encoding reuses the existing physical-reference evaluation
lemmas. Public mathematical/native formulas use logical channels directly. -/
def flatTensor {α : Type} (zero : α) (heads time width : Nat)
    (xs : TileIndex [heads,time,width] → α) (off : Nat) : α :=
  if h : off<heads*time*width then
    let outer : Fin (heads*time) × Fin width := finProdFinEquiv.symm ⟨off,h⟩
    let inner : Fin heads × Fin time := finProdFinEquiv.symm outer.1
    xs (inner.1,inner.2,outer.2,PUnit.unit)
  else zero

theorem flatTensor_dense {α : Type} (zero : α) (heads time width : Nat)
    (xs : TileIndex [heads,time,width] → α) (bh t e : Nat)
    (hb : bh<heads) (ht : t<time) (he : e<width) :
    flatTensor zero heads time width xs (dense time width bh t e)=
      xs (⟨bh,hb⟩,⟨t,ht⟩,⟨e,he⟩,PUnit.unit) := by
  have bound := dense_bound time width heads bh t e hb ht he
  have encoded : (⟨dense time width bh t e,bound⟩ : Fin (heads*time*width))=
      finProdFinEquiv (finProdFinEquiv (⟨bh,hb⟩,⟨t,ht⟩),⟨e,he⟩) := by
    apply Fin.ext
    change (bh*time+t)*width+e=e+width*(t+time*bh)
    ring
  simp only [flatTensor,dif_pos bound,encoded,Equiv.symm_apply_apply]

def keyCount (s : Shape) : Nat := s.B*s.H*s.T*s.DK
def valueCount (s : Shape) : Nat := s.B*s.H*s.T*s.DV
def packedBases (s : Shape) : Bases :=
  ⟨0,keyCount s,keyCount s+keyCount s,keyCount s+keyCount s+valueCount s⟩

def packedMemory {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) (off : Nat) : α :=
  if off<keyCount s then flatTensor zero (s.B*s.H) s.T s.DK (xs (0 : Fin 4)) off
  else if off<keyCount s+keyCount s then
    flatTensor zero (s.B*s.H) s.T s.DK (xs (1 : Fin 4)) (off-keyCount s)
  else if off<keyCount s+keyCount s+valueCount s then
    flatTensor zero (s.B*s.H) s.T s.DV (xs (2 : Fin 4)) (off-(keyCount s+keyCount s))
  else flatTensor zero (s.B*s.H) s.T s.DV (xs (3 : Fin 4))
    (off-(keyCount s+keyCount s+valueCount s))

theorem packed_query {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.DK) :
    packedMemory zero s xs ((packedBases s).q+dense s.T s.DK bh t e)=
      xs (0 : Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨e,he⟩,PUnit.unit) := by
  have bound : dense s.T s.DK bh t e<keyCount s := dense_bound s.T s.DK (s.B*s.H) bh t e hb ht he
  simpa only [packedMemory,packedBases,Nat.zero_add,if_pos bound] using
    flatTensor_dense zero (s.B*s.H) s.T s.DK (xs (0 : Fin 4)) bh t e hb ht he

theorem packed_key {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.DK) :
    packedMemory zero s xs ((packedBases s).k+dense s.T s.DK bh t e)=
      xs (1 : Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨e,he⟩,PUnit.unit) := by
  have bound : dense s.T s.DK bh t e<keyCount s := dense_bound s.T s.DK (s.B*s.H) bh t e hb ht he
  have first : ¬keyCount s+dense s.T s.DK bh t e<keyCount s := by omega
  have second : keyCount s+dense s.T s.DK bh t e<keyCount s+keyCount s := by omega
  simpa only [packedMemory,packedBases,if_neg first,if_pos second,Nat.add_sub_cancel_left] using
    flatTensor_dense zero (s.B*s.H) s.T s.DK (xs (1 : Fin 4)) bh t e hb ht he

theorem packed_value {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.DV) :
    packedMemory zero s xs ((packedBases s).v+dense s.T s.DV bh t e)=
      xs (2 : Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨e,he⟩,PUnit.unit) := by
  have bound : dense s.T s.DV bh t e<valueCount s := dense_bound s.T s.DV (s.B*s.H) bh t e hb ht he
  have first : ¬keyCount s+keyCount s+dense s.T s.DV bh t e<keyCount s := by omega
  have second : ¬keyCount s+keyCount s+dense s.T s.DV bh t e<keyCount s+keyCount s := by omega
  have third : keyCount s+keyCount s+dense s.T s.DV bh t e<keyCount s+keyCount s+valueCount s := by omega
  simpa only [packedMemory,packedBases,if_neg first,if_neg second,if_pos third,Nat.add_sub_cancel_left] using
    flatTensor_dense zero (s.B*s.H) s.T s.DV (xs (2 : Fin 4)) bh t e hb ht he

theorem packed_gradient {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.DV) :
    packedMemory zero s xs ((packedBases s).g+dense s.T s.DV bh t e)=
      xs (3 : Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨e,he⟩,PUnit.unit) := by
  have first : ¬keyCount s+keyCount s+valueCount s+dense s.T s.DV bh t e<keyCount s := by omega
  have second : ¬keyCount s+keyCount s+valueCount s+dense s.T s.DV bh t e<keyCount s+keyCount s := by omega
  have third : ¬keyCount s+keyCount s+valueCount s+dense s.T s.DV bh t e<keyCount s+keyCount s+valueCount s := by omega
  simpa only [packedMemory,packedBases,if_neg first,if_neg second,if_neg third,Nat.add_sub_cancel_left] using
    flatTensor_dense zero (s.B*s.H) s.T s.DV (xs (3 : Fin 4)) bh t e hb ht he

theorem inputs_ext {α : Type} (x y : Inputs α)
    (hq : x.q=y.q) (hk : x.k=y.k) (hv : x.v=y.v) (hg : x.g=y.g) : x=y := by
  cases x
  cases y
  simp_all

theorem packed_inputs {α : Type} (a : Ops α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) :
    currentInputs a s (packedBases s) (packedMemory a.zero s xs)=backwardLogical a.zero s xs := by
  apply inputs_ext
  all_goals
    funext bh t e
    simp only [currentInputs,tensorRead,backwardLogical,tensor]
    split_ifs with h
  · exact packed_query a.zero s xs bh t e h.1 h.2.1 h.2.2
  · rfl
  · exact packed_key a.zero s xs bh t e h.1 h.2.1 h.2.2
  · rfl
  · exact packed_value a.zero s xs bh t e h.1 h.2.1 h.2.2
  · rfl
  · exact packed_gradient a.zero s xs bh t e h.1 h.2.1 h.2.2
  · rfl

def extendForward {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (forwardInputs s)) : ChannelValues α (backwardInputs s) :=
  Fin.cases (xs (0 : Fin 3)) (Fin.cases (xs (1 : Fin 3))
    (Fin.cases (xs (2 : Fin 3)) (Fin.cases (fun _ => zero) (fun k => k.elim0))))

theorem extendForward_logical {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (forwardInputs s)) :
    backwardLogical zero s (extendForward zero s xs)=forwardLogical zero s xs := by
  apply inputs_ext
  · rfl
  · rfl
  · rfl
  · funext bh t e
    change tensor zero (s.B*s.H) s.T s.DV (fun _ => zero) bh t e=zero
    simp [tensor]

#axiomsClean packed_inputs
#axiomsClean extendForward_logical
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
