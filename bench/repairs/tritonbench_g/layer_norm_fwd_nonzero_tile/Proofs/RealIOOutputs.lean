import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RealIOInputs
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.ActiveGrid

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {f : Flags} {r : Regions} {stride : Strides} {M N : Nat} {eps p : ℝ}

def matrixRow (j : Fin (M*N)) : Fin M :=
  ⟨j.val/N,(Nat.div_lt_iff_lt_mul (by have:=j.isLt; nlinarith)).mpr j.isLt⟩
def matrixCol (j : Fin (M*N)) : Fin N :=
  ⟨j.val%N,Nat.mod_lt _ (by have:=j.isLt; nlinarith)⟩

theorem output_binding (a : Arrays M N) (s t : BlockState)
    (hin : (io f r stride M N eps p).InputBinding s a)
    (geometry : NativeGeometry.Geometry f r stride M N)
    (post : WholeGrid.Post f r stride M N eps p s t) :
    (io f r stride M N eps p).OutputBinding s t a (formula f M N eps p a) := by
  have read_real {region : RegionName} {off : Nat} {value : ℝ}
      (h : t.mem region off=MemCell.real value) : t.readMem region off=value := by
    simp only [BlockState.readMem,h,MemCell.readAs_real_real]
  have read_bool {region : RegionName} {off : Nat} {value : Bool}
      (h : t.mem region off=MemCell.of .bool value) : t.readMemValue .bool region off=value := by
    simp only [BlockState.readMemValue,BlockState.readMemTyped,h,MemCell.readAs_of_same]
  have agree:=input_agreement a s hin
  intro o j on
  fin_cases o
  · change t.readMem r.y (j.val/N*stride.y+j.val%N)=_
    have result:=RealRevision.ActiveGrid.y_cell f r stride M N eps p s t geometry post (matrixRow j) (matrixCol j)
    exact (read_real result).trans (agree.y eps p (matrixRow j).val (matrixCol j).val (matrixRow j).isLt (matrixCol j).isLt)
  · change t.readMem r.y1 (j.val/N*stride.y1+j.val%N)=_
    have result:=RealRevision.ActiveGrid.y1_cell f r stride M N eps p s t post on (matrixRow j) (matrixCol j)
    exact (read_real result).trans (agree.y1 eps p (matrixRow j).val (matrixCol j).val (matrixRow j).isLt (matrixCol j).isLt on)
  · change t.readMem r.mean j.val=_
    have result:=RealRevision.ActiveGrid.mean_cell f r stride M N eps p s t geometry post on j
    exact (read_real result).trans (agree.mean p j.val j.isLt)
  · change t.readMem r.rstd j.val=_
    have result:=RealRevision.ActiveGrid.rstd_cell f r stride M N eps p s t geometry post j
    exact (read_real result).trans (agree.rstd eps p j.val j.isLt)
  · change t.readMem r.residualOut (j.val/N*stride.residualOut+j.val%N)=_
    have result:=RealRevision.ActiveGrid.residual_cell f r stride M N eps p s t geometry post on (matrixRow j) (matrixCol j)
    exact (read_real result).trans (agree.combined p (matrixRow j).val (matrixCol j).val (matrixRow j).isLt (matrixCol j).isLt)
  · change t.readMemValue .bool r.dropoutMask j.val=_
    have result:=RealRevision.ActiveGrid.mask_cell f r stride M N eps p s t geometry post on.1 on.2 (matrixRow j) (matrixCol j)
    have address:(matrixRow j).val*N+(matrixCol j).val=j.val:=by
      simpa only [matrixRow,matrixCol,Nat.mul_comm] using Nat.div_add_mod j.val N
    rw [address] at result
    rw [read_bool result]
    exact congrArg (fun seed=>Formula.keep seed p (matrixCol j).val) (agree.seeds on.1 _ (matrixRow j).isLt)
  · change t.readMemValue .bool r.dropoutMask (M*N+j.val)=_
    have result:=RealRevision.ActiveGrid.mask1_cell f r stride M N eps p s t geometry post on.1 on.2.1 on.2.2 (matrixRow j) (matrixCol j)
    have address:(M+(matrixRow j).val)*N+(matrixCol j).val=M*N+j.val:=by
      simp only [Nat.add_mul,Nat.add_assoc,matrixRow,matrixCol]
      rw [Nat.mul_comm (j.val/N) N,Nat.div_add_mod]
    rw [address] at result
    rw [read_bool result]
    exact congrArg (fun seed=>Formula.keep seed p (matrixCol j).val) (agree.seeds1 on.2.2 on.1 _ (matrixRow j).isLt)

/-- Every actual written row cell is one visible active IO output. -/
theorem written_output (row : Fin M) (region : RegionName) (off : Nat)
    (written : Contract.written f r stride M N row.val region off) :
    ∃ (o : Fin 7) (j : Fin (outputArity M N o)),
      outputActive f o ∧ region=outputRegion r o ∧ off=outputAddress stride M N o j.val := by
  rcases written with ⟨active,region,address⟩ | ⟨region,address⟩ | ⟨region,address⟩ |
    ⟨active,region,address⟩ | ⟨active,region,address⟩ |
    ⟨drop,mask,x1,region,address⟩ | ⟨drop,mask,region,address⟩
  · obtain ⟨col,hc,eq⟩:=Contract.inRow_iff_col.mp address
    exact ⟨1,index row ⟨col,hc⟩,active,region,by simpa [outputAddress] using eq⟩
  · obtain ⟨col,hc,eq⟩:=Contract.inRow_iff_col.mp address
    exact ⟨0,index row ⟨col,hc⟩,trivial,region,by simpa [outputAddress] using eq⟩
  · exact ⟨3,row,trivial,region,address⟩
  · exact ⟨2,row,active,region,address⟩
  · obtain ⟨col,hc,eq⟩:=Contract.inRow_iff_col.mp address
    exact ⟨4,index row ⟨col,hc⟩,active,region,by simpa [outputAddress] using eq⟩
  · obtain ⟨col,hc,eq⟩:=Contract.inRow_iff_col.mp address
    exact ⟨6,index row ⟨col,hc⟩,⟨drop,mask,x1⟩,region,by simpa [outputAddress,index,Nat.add_mul,Nat.add_assoc] using eq⟩
  · obtain ⟨col,hc,eq⟩:=Contract.inRow_iff_col.mp address
    exact ⟨5,index row ⟨col,hc⟩,⟨drop,mask⟩,region,eq⟩

theorem frame (a : Arrays M N) (s t : BlockState)
    (post : WholeGrid.Post f r stride M N eps p s t) :
    (io f r stride M N eps p).Frame s t a := by
  intro region off outputs scratch
  apply post.2
  intro row hr hit
  obtain ⟨o,j,on,eq,address⟩:=written_output ⟨row,hr⟩ region off hit
  exact (outputs o j on).elim (fun h=>h eq) (fun h=>h address)

#axiomsClean output_binding
#axiomsClean written_output
#axiomsClean frame
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
