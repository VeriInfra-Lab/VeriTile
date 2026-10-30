import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeAgreement

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open RealInterface (index flat_lt flat_div index_div index_mod)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]
variable {f : Flags} {r : Regions} {stride : Strides} {M N : Nat} {eps p : ℝ}

theorem input_agreement (A : Algebra α) (a : Arrays r M N α) (s : State α)
    (hin : (io A f r stride M N eps p).InputBinding a s) :
    Agree f M N (snapshot r stride s) (logicalInputs a) := by
  constructor
  · intro i hi j hj
    have h := hin (0 : Fin 9) (index ⟨i,hi⟩ ⟨j,hj⟩,()) trivial
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,flat_lt hi hj,io,inputPorts,mapValue,
      inputIndex,RealInterface.inputAddress,index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,io,inputPorts,mapValue] using h
  · intro hf i hi j hj
    have h := hin (1 : Fin 9) (index ⟨i,hi⟩ ⟨j,hj⟩,()) hf
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,flat_lt hi hj,io,inputPorts,mapValue,
      inputIndex,RealInterface.inputAddress,index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,io,inputPorts,mapValue] using h
  · intro hf i hi j hj
    have h := hin (2 : Fin 9) (index ⟨i,hi⟩ ⟨j,hj⟩,()) hf
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,flat_lt hi hj,io,inputPorts,mapValue,
      inputIndex,RealInterface.inputAddress,index,flat_div i j hj,Nat.add_mod,Nat.mod_eq_of_lt hj,io,inputPorts,mapValue] using h
  · intro j hj
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,hj,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (3 : Fin 9) (⟨j,hj⟩,()) trivial
  · intro hf j hj
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,hj,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (4 : Fin 9) (⟨j,hj⟩,()) hf
  · intro hf j hj
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,hj,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (5 : Fin 9) (⟨j,hj⟩,()) hf
  · intro hf hb j hj
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,hj,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (6 : Fin 9) (⟨j,hj⟩,()) ⟨hf,hb⟩
  · intro hf i hi
    have h2 : i<2*M := by dsimp [RealInterface.inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,h2,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (7 : Fin 9) (⟨i,h2⟩,()) ⟨hf,Or.inl hi⟩
  · intro hx hf i hi
    have h2 : M+i<2*M := by dsimp [RealInterface.inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,h2,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (7 : Fin 9) (⟨M+i,h2⟩,()) ⟨hf,Or.inr hx⟩
  · intro hf i hi
    have h2 : i<2*M := by dsimp [RealInterface.inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,h2,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (8 : Fin 9) (⟨i,h2⟩,()) ⟨hf,Or.inl hi⟩
  · intro hx hf i hi
    have h2 : M+i<2*M := by dsimp [RealInterface.inputArity]; omega
    simpa [snapshot,logicalInputs,lookup,RealInterface.inputArity,h2,io,inputPorts,mapValue,io,inputPorts,mapValue]
      using hin (8 : Fin 9) (⟨M+i,h2⟩,()) ⟨hf,Or.inr hx⟩

theorem row_bounds (A : Algebra α) (a : Arrays r M N α) (bounds : RegionBounds)
    (hb : (io A f r stride M N eps p).InBounds bounds a) (row : Fin M) :
    FlatDomain.RowBounds f r stride M N row.val bounds := by
  have ins (i : Fin 9) (j : Fin (RealInterface.inputArity M N i))
      (active : RealInterface.inputActive f M i j.val) :
      RealInterface.inputAddress stride N i j.val<bounds (RealInterface.inputRegion r i) := by
    fin_cases i
    · exact hb.2.1 (0 : Fin 9) (j,()) active
    · exact hb.2.1 (1 : Fin 9) (j,()) active
    · exact hb.2.1 (2 : Fin 9) (j,()) active
    · exact hb.2.1 (3 : Fin 9) (j,()) active
    · exact hb.2.1 (4 : Fin 9) (j,()) active
    · exact hb.2.1 (5 : Fin 9) (j,()) active
    · exact hb.2.1 (6 : Fin 9) (j,()) active
    · exact hb.2.1 (7 : Fin 9) (j,()) active
    · exact hb.2.1 (8 : Fin 9) (j,()) active
  have outs (o : Fin 7) (j : Fin (RealInterface.outputArity M N o))
      (active : RealInterface.outputActive f o) :
      RealInterface.outputAddress stride M N o j.val<bounds (RealInterface.outputRegion r o) := by
    fin_cases o
    · exact hb.2.2.1 (0 : Fin 7) (j,()) active
    · exact hb.2.2.1 (1 : Fin 7) (j,()) active
    · exact hb.2.2.1 (2 : Fin 7) (j,()) active
    · exact hb.2.2.1 (3 : Fin 7) (j,()) active
    · exact hb.2.2.1 (4 : Fin 7) (j,()) active
    · exact hb.2.2.1 (5 : Fin 7) (j,()) active
    · exact hb.2.2.1 (6 : Fin 7) (j,()) active
  constructor
  · intro j hj
    simpa [RealInterface.inputAddress,RealInterface.inputRegion] using ins 0 (index row ⟨j,hj⟩) trivial
  · intro j hj
    simpa [RealInterface.outputAddress,RealInterface.outputRegion] using outs 0 (index row ⟨j,hj⟩) trivial
  · intro j hj
    exact ins 3 ⟨j,hj⟩ trivial
  · exact outs 3 row trivial
  · intro hf j hj
    exact ins 4 ⟨j,hj⟩ hf
  · intro hf
    exact outs 2 row hf
  · intro hf j hj
    simpa [RealInterface.inputAddress,RealInterface.inputRegion] using ins 2 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    simpa [RealInterface.outputAddress,RealInterface.outputRegion] using outs 4 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    simpa [RealInterface.inputAddress,RealInterface.inputRegion] using ins 1 (index row ⟨j,hj⟩) hf
  · intro hf j hj
    exact ins 5 ⟨j,hj⟩ hf
  · intro hf hf' j hj
    exact ins 6 ⟨j,hj⟩ ⟨hf,hf'⟩
  · intro hf j hj
    simpa [RealInterface.outputAddress,RealInterface.outputRegion] using outs 1 (index row ⟨j,hj⟩) hf
  · intro hf
    exact ins 7 ⟨row.val,by dsimp [RealInterface.inputArity]; omega⟩ ⟨hf,Or.inl row.isLt⟩
  · intro hx hf
    exact ins 7 ⟨M+row.val,by dsimp [RealInterface.inputArity]; omega⟩ ⟨hf,Or.inr hx⟩
  · intro hf
    exact ins 8 ⟨row.val,by dsimp [RealInterface.inputArity]; omega⟩ ⟨hf,Or.inl row.isLt⟩
  · intro hx hf
    exact ins 8 ⟨M+row.val,by dsimp [RealInterface.inputArity]; omega⟩ ⟨hf,Or.inr hx⟩
  · intro hd hm j hj
    exact outs 5 (index row ⟨j,hj⟩) ⟨hd,hm⟩
  · intro hx hd hm j hj
    simpa [RealInterface.outputAddress,RealInterface.outputRegion,index,Nat.add_mul,Nat.add_assoc] using
      outs 6 (index row ⟨j,hj⟩) ⟨hd,hm,hx⟩

#axiomsClean input_agreement
#axiomsClean row_bounds
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
