import bench.tritonbench_g.layer_norm_fwd.Proofs.NativeIOInputs
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealIOOutputs

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open RealInterface (matrixRow matrixCol)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]
variable {f : Flags} {r : Regions} {stride : Strides} {M N : Nat} {eps p : ℝ}

theorem output_binding (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (width : N≤tile N) (a : Arrays r M N α) (s t : State α)
    (hin : (io A f r stride M N eps p).InputBinding a s)
    (post : TypedSource.GridPost A cmp f r stride M N eps p (tile N) s t) :
    (io A f r stride M N eps p).OutputBinding a (target A cmp f r stride M N eps p width a) t := by
  have agree:=input_agreement A a s hin
  have values (row : Fin M) : TypedSource.combinedValues A cmp f s r stride M N p row.val (tile N)=
      combined A cmp f (logicalInputs a) M N p row.val (tile N) :=
    (combined_snapshot A cmp f r stride M N p row.val (tile N) s).trans
      (agree.combined A cmp p row.val row.isLt (tile N))
  have load {region : RegionName} {expected : Nat → α} (col : Fin N)
      (bound : (s.mem region col.val).read .real=expected col.val) :
      TypedSource.raw A s region 0 N col.val=loaded A N expected col.val := by
    simp only [TypedSource.raw,loaded,TypedSource.cast,Nat.zero_add,if_pos col.isLt,bound]
  have mapped (cell : Cell α) : mapCell id cell=cell := by
    cases cell with | mk d val=>cases d <;> rfl
  intro port j on
  fin_cases port
  · change mapCell id (t.mem r.y (j.1.val/N*stride.y+j.1.val%N))=_
    have result:=(post.1 (matrixRow j.1)).2.1 (TypedSource.lane width (matrixCol j.1)) (matrixCol j.1).isLt
    dsimp only [matrixRow,matrixCol,TypedSource.lane] at result
    rw [mapped,result]
    have weight:=load (matrixCol j.1) (agree.weight _ (matrixCol j.1).isLt)
    have bias (hf : f.bias=Bool.true):=load (matrixCol j.1) (agree.bias hf _ (matrixCol j.1).isLt)
    have v:=values (matrixRow j.1)
    dsimp only [matrixRow] at v
    dsimp only [matrixCol] at weight bias
    rw [v,weight]
    cases hf:f.bias <;> simp [target,TypedSource.affineValue,hf,bias,io,outputPorts] <;> rfl
  · change mapCell id (t.mem r.y1 (j.1.val/N*stride.y1+j.1.val%N))=_
    have result:=(post.1 (matrixRow j.1)).2.2.1 on (TypedSource.lane width (matrixCol j.1)) (matrixCol j.1).isLt
    dsimp only [matrixRow,matrixCol,TypedSource.lane] at result
    rw [mapped,result]
    have weight:=load (matrixCol j.1) (agree.weight1 on _ (matrixCol j.1).isLt)
    have bias (hf : f.b1=Bool.true):=load (matrixCol j.1) (agree.bias1 on hf _ (matrixCol j.1).isLt)
    have v:=values (matrixRow j.1)
    dsimp only [matrixRow] at v
    dsimp only [matrixCol] at weight bias
    rw [v,weight]
    cases hf:f.b1 <;> simp [target,TypedSource.affineValue,hf,bias,io,outputPorts] <;> rfl
  · change mapCell id (t.mem r.mean j.1.val)=_
    rw [mapped,(post.1 j.1).2.2.2.2 on,values]
    rfl
  · change mapCell id (t.mem r.rstd j.1.val)=_
    rw [mapped,(post.1 j.1).2.2.2.1,values]
    rfl
  · change mapCell id (t.mem r.residualOut (j.1.val/N*stride.residualOut+j.1.val%N))=_
    have result:=(post.1 (matrixRow j.1)).1.1 on (TypedSource.lane width (matrixCol j.1)) (matrixCol j.1).isLt
    dsimp only [matrixRow,matrixCol,TypedSource.lane] at result
    have v:=values (matrixRow j.1)
    dsimp only [matrixRow] at v
    rw [mapped,result,v]
    rfl
  · change mapCell id (t.mem r.dropoutMask j.1.val)=_
    have address:(matrixRow j.1).val*N+(matrixCol j.1).val=j.1.val:=by
      simpa only [matrixRow,matrixCol,Nat.mul_comm] using Nat.div_add_mod j.1.val N
    have result:=(post.1 (matrixRow j.1)).1.2.1 on.1 on.2 (TypedSource.lane width (matrixCol j.1)) (matrixCol j.1).isLt
    change t.mem r.dropoutMask ((matrixRow j.1).val*N+(matrixCol j.1).val)=_ at result
    rw [address] at result
    rw [mapped,result]
    have seed:=agree.seeds on.1 _ (matrixRow j.1).isLt
    change (s.mem r.seeds (matrixRow j.1).val).read .int=_ at seed
    simp only [seed]
    rfl
  · change mapCell id (t.mem r.dropoutMask (M*N+j.1.val))=_
    have address:(M+(matrixRow j.1).val)*N+(matrixCol j.1).val=M*N+j.1.val:=by
      simp only [Nat.add_mul,Nat.add_assoc,matrixRow,matrixCol]
      rw [Nat.mul_comm (j.1.val/N) N,Nat.div_add_mod]
    have result:=(post.1 (matrixRow j.1)).1.2.2 on.1 on.2.1 on.2.2
      (TypedSource.lane width (matrixCol j.1)) (matrixCol j.1).isLt
    change t.mem r.dropoutMask ((M+(matrixRow j.1).val)*N+(matrixCol j.1).val)=_ at result
    rw [address] at result
    rw [mapped,result]
    have seed:=agree.seeds1 on.2.2 on.1 _ (matrixRow j.1).isLt
    change (s.mem r.seeds (M+(matrixRow j.1).val)).read .int=_ at seed
    simp only [seed]
    rfl

theorem frame (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (a : Arrays r M N α) (s t : State α)
    (post : TypedSource.GridPost A cmp f r stride M N eps p (tile N) s t) :
    (io A f r stride M N eps p).Frame a s t := by
  intro region off outside
  apply post.2
  intro row hit
  obtain ⟨o,j,on,eq,address⟩:=RealInterface.written_output row region off hit
  have miss : region≠RealInterface.outputRegion r o ∨ off≠RealInterface.outputAddress stride M N o j.val := by
    fin_cases o
    all_goals first
    | exact outside.1 (0 : Fin 7) (j,()) on
    | exact outside.1 (1 : Fin 7) (j,()) on
    | exact outside.1 (2 : Fin 7) (j,()) on
    | exact outside.1 (3 : Fin 7) (j,()) on
    | exact outside.1 (4 : Fin 7) (j,()) on
    | exact outside.1 (5 : Fin 7) (j,()) on
    | exact outside.1 (6 : Fin 7) (j,()) on
  exact miss.elim (fun h=>h eq) (fun h=>h address)

#axiomsClean output_binding
#axiomsClean frame
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
