import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.IO
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.LogicalAgreement
import bench.tritonbench_g.layer_norm_fwd.Proofs.FlatDomain

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000

theorem flat_lt {M N i j : Nat} (hi : i<M) (hj : j<N) : i*N+j<M*N := by
  nlinarith

theorem flat_div (i j : Nat) {N : Nat} (hj : j<N) : (i*N+j)/N=i := by
  rw [Nat.mul_comm i N,Nat.mul_add_div (Nat.zero_lt_of_lt hj),Nat.div_eq_of_lt hj,Nat.add_zero]

def index {M N : Nat} (i : Fin M) (j : Fin N) : Fin (M*N) :=
  ⟨i.val*N+j.val,flat_lt i.isLt j.isLt⟩

@[simp] theorem index_div {M N : Nat} (i : Fin M) (j : Fin N) :
    (index i j).val/N=i.val := by
  exact flat_div i.val j.val j.isLt

@[simp] theorem index_mod {M N : Nat} (i : Fin M) (j : Fin N) :
    (index i j).val%N=j.val := by
  simp [index,Nat.add_mod,Nat.mod_eq_of_lt j.isLt]

variable {f : Flags} {r : Regions} {stride : Strides} {M N : Nat} {eps p : ℝ}

/-- Simultaneous typed bindings preserve all original input aliases and
wrong-tag defaults; only enabled logical lanes are used by the formula. -/
theorem input_agreement (a : Arrays M N) (s : BlockState)
    (hin : (io f r stride M N eps p).InputBinding s a) :
    LogicalAgreement.Agree f M N (snapshot r stride s) (logicalInputs M N a) := by
  have bound (i : Fin 9) (j : Fin (inputArity M N i))
      (active : inputActive f M i j.val) :
      (inputType i).read s (inputRegion r i) (inputAddress stride N i j.val)=a i j :=
    hin i j active
  constructor
  · intro i hi j hj
    have h := bound 0 (index ⟨i,hi⟩ ⟨j,hj⟩) trivial
    simpa [snapshot,logicalInputs,lookup,inputArity,flat_lt hi hj,inputType,inputRegion,inputAddress,
      index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,ChanTy.read] using h
  · intro hf i hi j hj
    have h := bound 1 (index ⟨i,hi⟩ ⟨j,hj⟩) hf
    simpa [snapshot,logicalInputs,lookup,inputArity,flat_lt hi hj,inputType,inputRegion,inputAddress,
      index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,ChanTy.read] using h
  · intro hf i hi j hj
    have h := bound 2 (index ⟨i,hi⟩ ⟨j,hj⟩) hf
    simpa [snapshot,logicalInputs,lookup,inputArity,flat_lt hi hj,inputType,inputRegion,inputAddress,
      index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,ChanTy.read] using h
  · intro j hj
    simpa [snapshot,logicalInputs,lookup,inputArity,hj,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 3 ⟨j,hj⟩ trivial
  · intro hf j hj
    simpa [snapshot,logicalInputs,lookup,inputArity,hj,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 4 ⟨j,hj⟩ hf
  · intro hf j hj
    simpa [snapshot,logicalInputs,lookup,inputArity,hj,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 5 ⟨j,hj⟩ hf
  · intro hf hb j hj
    simpa [snapshot,logicalInputs,lookup,inputArity,hj,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 6 ⟨j,hj⟩ ⟨hf,hb⟩
  · intro hf i hi
    have h2 : i<2*M := by dsimp [inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,inputArity,h2,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 7 ⟨i,h2⟩ ⟨hf,Or.inl hi⟩
  · intro hx hf i hi
    have h2 : M+i<2*M := by dsimp [inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,inputArity,h2,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 7 ⟨M+i,h2⟩ ⟨hf,Or.inr hx⟩
  · intro hf i hi
    have h2 : i<2*M := by dsimp [inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,inputArity,h2,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 8 ⟨i,h2⟩ ⟨hf,Or.inl hi⟩
  · intro hx hf i hi
    have h2 : M+i<2*M := by dsimp [inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,inputArity,h2,inputType,inputRegion,inputAddress,ChanTy.read]
      using bound 8 ⟨M+i,h2⟩ ⟨hf,Or.inr hx⟩

/-- Each bound is the actual source row/column address, including both seed,
row-scale and mask streams. No unused row gaps or disabled buffers are assumed. -/
theorem row_bounds (a : Arrays M N) (s : BlockState) (bounds : RegionBounds)
    (hb : (io f r stride M N eps p).InBounds bounds s a) (row : Fin M) :
    FlatDomain.RowBounds f r stride M N row.val bounds := by
  have ins (i : Fin 9) (j : Fin (inputArity M N i))
      (active : inputActive f M i j.val) :
      inputAddress stride N i j.val<bounds (inputRegion r i) := hb.1 i j active
  have outs (o : Fin 7) (j : Fin (outputArity M N o)) (active : outputActive f o) :
      outputAddress stride M N o j.val<bounds (outputRegion r o) := hb.2.1 o j active
  constructor
  · intro j hj
    simpa [inputAddress,inputRegion] using ins 0 (index row ⟨j,hj⟩) trivial
  · intro j hj
    simpa [outputAddress,outputRegion] using outs 0 (index row ⟨j,hj⟩) trivial
  · intro j hj
    exact ins 3 ⟨j,hj⟩ trivial
  · exact outs 3 row trivial
  · intro hf j hj
    exact ins 4 ⟨j,hj⟩ hf
  · intro hf
    exact outs 2 row hf
  · intro hf j hj
    simpa [inputAddress,inputRegion] using ins 2 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    simpa [outputAddress,outputRegion] using outs 4 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    simpa [inputAddress,inputRegion] using ins 1 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    exact ins 5 ⟨j,hj⟩ hf
  · intro hf hf' j hj
    exact ins 6 ⟨j,hj⟩ ⟨hf,hf'⟩
  · intro hf j hj
    simpa [outputAddress,outputRegion] using outs 1 (index row ⟨j,hj⟩) hf
  · intro hf
    exact ins 7 ⟨row.val,by dsimp [inputArity]; omega⟩ ⟨hf,Or.inl row.isLt⟩
  · intro hx hf
    exact ins 7 ⟨M+row.val,by dsimp [inputArity]; omega⟩ ⟨hf,Or.inr hx⟩
  · intro hf
    exact ins 8 ⟨row.val,by dsimp [inputArity]; omega⟩ ⟨hf,Or.inl row.isLt⟩
  · intro hx hf
    exact ins 8 ⟨M+row.val,by dsimp [inputArity]; omega⟩ ⟨hf,Or.inr hx⟩
  · intro hd hm j hj
    exact outs 5 (index row ⟨j,hj⟩) ⟨hd,hm⟩
  · intro hx hd hm j hj
    simpa [outputAddress,outputRegion,index,Nat.add_mul,Nat.add_assoc] using
      outs 6 (index row ⟨j,hj⟩) ⟨hd,hm,hx⟩

#axiomsClean input_agreement
#axiomsClean row_bounds
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
