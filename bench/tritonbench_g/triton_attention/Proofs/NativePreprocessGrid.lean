import bench.tritonbench_g.triton_attention.Proofs.NativePreprocessValues
import bench.tritonbench_g.triton_attention.Proofs.NativeBackwardGrid
import bench.tritonbench_g.triton_attention.Proofs.PreGridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000

def flatTensor (c : Shape) (f : Target.Tensor) (row d : Nat) : Graph:=f (row/c.rows) (row%c.rows) d
def flatRows (c : Shape) (f : Target.Rows) (row : Nat) : Graph:=f (row/c.rows) (row%c.rows)
structure PreData (c : Shape) (mem : Grid.Memory Graph) (x : Target.Inputs) (saved : Target.Saved) : Prop where
  output : TensorData c mem "Out" saved.output
  gradient : TensorData c mem "DO" x.grad
  denominator : RowData c mem "L" saved.denominator

theorem flatTensor_at (c : Shape) (f : Target.Tensor) (h : Fin c.groups) (i : Fin c.rows) (d : Nat) :
    flatTensor c f (rowOffset c h i) d=f h.val i.val d := by
  have hd : (h.val*c.rows+i.val)/c.rows=h.val := by
    rw [Nat.mul_comm h.val c.rows,Nat.mul_add_div c.rows_pos,Nat.div_eq_of_lt i.isLt,Nat.add_zero]
  simp [flatTensor,rowOffset,hd,Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
theorem flatRows_at (c : Shape) (f : Target.Rows) (h : Fin c.groups) (i : Fin c.rows) :
    flatRows c f (rowOffset c h i)=f h.val i.val := by
  have hd : (h.val*c.rows+i.val)/c.rows=h.val := by
    rw [Nat.mul_comm h.val c.rows,Nat.mul_add_div c.rows_pos,Nat.div_eq_of_lt i.isLt,Nat.add_zero]
  simp [flatRows,rowOffset,hd,Nat.add_mod,Nat.mod_eq_of_lt i.isLt]

theorem tensorData_flat (c : Shape) (mem : Grid.Memory Graph) (r : RegionName) (f : Target.Tensor)
    (h : TensorData c mem r f) (row : Nat) (hr : row<c.groups*c.rows) (d : Fin c.dim) :
    mem r (row*c.dim+d.val)=.mk .fp32 (flatTensor c f row d.val) := by
  have hd : row/c.rows<c.groups:=Nat.div_lt_iff_lt_mul c.rows_pos |>.mpr hr
  have ans:=h (⟨row/c.rows,hd⟩,⟨row%c.rows,Nat.mod_lt _ c.rows_pos⟩,d)
  simpa [offset,flatTensor,Nat.div_add_mod'] using ans

theorem rowData_flat (c : Shape) (mem : Grid.Memory Graph) (r : RegionName) (f : Target.Rows)
    (h : RowData c mem r f) (row : Nat) (hr : row<c.groups*c.rows) :
    mem r row=.mk .fp32 (flatRows c f row) := by
  have hd : row/c.rows<c.groups:=Nat.div_lt_iff_lt_mul c.rows_pos |>.mpr hr
  have ans:=h ⟨row/c.rows,hd⟩ ⟨row%c.rows,Nat.mod_lt _ c.rows_pos⟩
  simpa [rowOffset,flatRows,Nat.div_add_mod'] using ans

theorem preData_ready (c : Shape) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (x : Target.Inputs) (saved : Target.Saved) (h : PreData c mem x saved)
    (hp : Grid.Legal (Grid.preprocessGrid c) pid) :
    PreReady c (Grid.entry (Grid.preprocessGrid c) pid (embed mem))
      (flatTensor c saved.output) (flatTensor c x.grad) (flatRows c saved.denominator) := by
  have tensors (r : RegionName) (f : Target.Tensor) (hf : TensorData c mem r f) (i : PreLane (preParameters c)) :
      (((Grid.entry (Grid.preprocessGrid c) pid (embed mem)).mem r
        (preOffset (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid (embed mem)) i)).read .fp32)=
          fun _=>Stage.ready (flatTensor c f (preRow (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid (embed mem)) (i.1,())) i.2.1.val) := by
    have hm:=tensorData_flat c mem r f hf _ (Grid.preRow_bound c pid (embed mem) hp (i.1,())) i.2.1
    change mem r (preOffset (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid (embed mem)) i)=_ at hm
    simp only [Grid.entry] at hm
    simp only [Grid.entry,embed,hm]
    rfl
  refine ⟨tensors "Out" saved.output h.output,tensors "DO" x.grad h.gradient,?_⟩
  intro i
  have hm:=rowData_flat c mem "L" saved.denominator h.denominator _ (Grid.preRow_bound c pid (embed mem) hp i)
  simp only [Grid.entry] at hm
  simp only [Grid.entry,embed,hm]
  rfl

theorem preprocessStep_readonly (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (r : RegionName) (o : Nat) (hr : r≠"NewDO" ∧ r≠"Delta") :
    store one .preprocess c scale owners pid mem r o=mem r o := by
  have hr' : r≠(preParameters c).newGradient ∧ r≠(preParameters c).delta:=hr
  change mapCell _ (Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) r o)=_
  unfold Grid.preprocessStep preWrite
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2),State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.1)]
  exact congrFun (congrFun (finish_embed one .preprocess c owners mem) r) o

theorem preprocessStep_data (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (x : Target.Inputs) (saved : Target.Saved) (h : PreData c mem x saved) :
    PreData c (store one .preprocess c scale owners pid mem) x saved := by
  constructor
  · intro i;rw [preprocessStep_readonly one c scale owners pid mem "Out" _ (by decide)];exact h.output i
  · intro i;rw [preprocessStep_readonly one c scale owners pid mem "DO" _ (by decide)];exact h.gradient i
  · intro hd i;rw [preprocessStep_readonly one c scale owners pid mem "L" _ (by decide)];exact h.denominator hd i

theorem storeOwner_delta (c : Shape) (owners : TritonAttention.Whole001.Native.Owners) (h : Fin c.groups) (i : Fin c.rows) :
    storeOwner c owners "Delta" (rowOffset c h i)=owners.delta h.val i.val%(c.dim/4) := by
  have hd : (h.val*c.rows+i.val)/c.rows=h.val := by
    rw [Nat.mul_comm h.val c.rows,Nat.mul_add_div c.rows_pos,Nat.div_eq_of_lt i.isLt,Nat.add_zero]
  simp [storeOwner,rowOffset,hd,Nat.add_mod,Nat.mod_eq_of_lt i.isLt]

theorem preprocessStep_values (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (mem : Grid.Memory Graph)
    (x : Target.Inputs) (saved : Target.Saved) (h : PreData c mem x saved) (i : Coord c) :
    store one .preprocess c scale owners (Grid.preProgram c i) mem "NewDO" (offset c i)=
      .mk .fp32 ((Target.preprocess c owners x saved).gradient i.1.val i.2.1.val i.2.2.val) ∧
    store one .preprocess c scale owners (Grid.preProgram c i) mem "Delta" (rowOffset c i.1 i.2.1)=
      .mk .fp32 ((Target.preprocess c owners x saved).delta i.1.val i.2.1.val) := by
  let pid:=Grid.preProgram c i
  let s:=Grid.entry (Grid.preprocessGrid c) pid (embed mem)
  have hp:=Grid.preProgram_legal c i
  have reads:=preData_ready c pid mem x saved h hp
  obtain ⟨final,_,hm,hg,hd,_⟩:=preprocessCorrectness (algebraScalar one .preprocess) (modelScalar one .preprocess)
    (preParameters c) s (by simp [preParameters]) (Grid.preOffset_injective c pid (embed mem))
  have a:=hg (Grid.preLane c i)
  have b:=hd ((Grid.preLane c i).1,())
  rw [hm] at a b
  change Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "NewDO" (preOffset (preParameters c) s (Grid.preLane c i))=_ at a
  change Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "Delta" (preRow (preParameters c) s ((Grid.preLane c i).1,()))=_ at b
  dsimp only [s,pid] at a b
  rw [Grid.pre_target_offset] at a
  rw [Grid.pre_target_row] at b
  change Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "NewDO" (offset c i)=
    Cell.mk .fp32 (preQuotient (modelScalar one .preprocess) (preParameters c) s (Grid.preLane c i)) at a
  change Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "Delta" (rowOffset c i.1 i.2.1)=
    Cell.mk .fp32 (preDelta (modelScalar one .preprocess) (preParameters c) s ((Grid.preLane c i).1,())) at b
  constructor
  · change mapCell _ (Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "NewDO" (offset c i))=_
    rw [a,preQuotient_native c one s _ _ _ reads]
    change Cell.mk .fp32 (Graph.div (flatTensor c x.grad (preRow (preParameters c) s ((Grid.preLane c i).1,())) i.2.2.val)
      (flatRows c saved.denominator (preRow (preParameters c) s ((Grid.preLane c i).1,()))))=_
    dsimp only [s,pid]
    rw [Grid.pre_target_row,flatTensor_at,flatRows_at]
    rfl
  · change mapCell _ (Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "Delta" (rowOffset c i.1 i.2.1))=_
    rw [b]
    change Cell.mk .fp32 (lowerScalar one .preprocess (preDelta (modelScalar one .preprocess) (preParameters c) s ((Grid.preLane c i).1,())
      (storeOwner c owners "Delta" (rowOffset c i.1 i.2.1))))=_
    rw [preDelta_native c one s _ _ _ reads,storeOwner_delta,Nat.mod_mod]
    dsimp only [s,pid]
    rw [Grid.pre_target_row]
    have hout : flatTensor c saved.output (rowOffset c i.1 i.2.1)=saved.output i.1.val i.2.1.val := by funext d;exact flatTensor_at c _ _ _ d
    have hgrad : flatTensor c x.grad (rowOffset c i.1 i.2.1)=x.grad i.1.val i.2.1.val := by funext d;exact flatTensor_at c _ _ _ d
    rw [hout,hgrad,flatRows_at]
    rfl

theorem preprocessStep_otherNewDO (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.preprocessGrid c) pid) (i : Coord c) (hn : pid≠Grid.preProgram c i) :
    store one .preprocess c scale owners pid mem "NewDO" (offset c i)=mem "NewDO" (offset c i) := by
  change mapCell _ (Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "NewDO" (offset c i))=_
  rw [Grid.preprocessStep_otherNewDO _ c pid (embed mem) hp i hn]
  exact congrFun (congrFun (finish_embed one .preprocess c owners mem) "NewDO") (offset c i)

theorem preprocessStep_otherDelta (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.preprocessGrid c) pid) (i : Coord c) (hn : pid≠Grid.preProgram c i) :
    store one .preprocess c scale owners pid mem "Delta" (rowOffset c i.1 i.2.1)=mem "Delta" (rowOffset c i.1 i.2.1) := by
  change mapCell _ (Grid.preprocessStep (modelScalar one .preprocess) c pid (embed mem) "Delta" (rowOffset c i.1 i.2.1))=_
  rw [Grid.preprocessStep_otherDelta _ c pid (embed mem) hp i hn]
  exact congrFun (congrFun (finish_embed one .preprocess c owners mem) "Delta") (rowOffset c i.1 i.2.1)

theorem preprocess_grid (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.preprocessGrid c) work) (mem : Grid.Memory Graph)
    (x : Target.Inputs) (saved : Target.Saved) (h : PreData c mem x saved) :
    ∃ final,Grid.run (execute one .preprocess c scale owners) work mem=some final ∧ PreData c final x saved ∧
      ∀ i : Coord c,
        final "NewDO" (offset c i)=.mk .fp32 ((Target.preprocess c owners x saved).gradient i.1.val i.2.1.val i.2.2.val) ∧
        final "Delta" (rowOffset c i.1 i.2.1)=.mk .fp32 ((Target.preprocess c owners x saved).delta i.1.val i.2.1.val) := by
  let final:=work.foldl (fun m pid=>store one .preprocess c scale owners pid m) mem
  have hinv : PreData c final x saved :=
    Grid.fold_invariant (store one .preprocess c scale owners) (fun m=>PreData c m x saved) work mem h
      (by intro pid hp m hm;exact preprocessStep_data one c scale owners pid m x saved hm)
  refine ⟨final,Grid.run_eq_fold _ _ (execute_success one .preprocess c scale owners) work mem,hinv,?_⟩
  intro i
  constructor
  · exact Grid.fold_readback (store one .preprocess c scale owners) (fun m=>PreData c m x saved) work mem
      (Grid.preProgram c i) "NewDO" (offset c i) _
      (Grid.FullSchedule.covers _ work schedule _ (Grid.preProgram_legal c i)) h
      (by intro pid hp m hm;exact preprocessStep_data one c scale owners pid m x saved hm)
      (by intro m hm;exact (preprocessStep_values one c scale owners m x saved hm i).1)
      (by intro pid hp hn m;exact preprocessStep_otherNewDO one c scale owners pid m (Grid.FullSchedule.legal _ work schedule pid hp) i hn)
  · exact Grid.fold_readback (store one .preprocess c scale owners) (fun m=>PreData c m x saved) work mem
      (Grid.preProgram c i) "Delta" (rowOffset c i.1 i.2.1) _
      (Grid.FullSchedule.covers _ work schedule _ (Grid.preProgram_legal c i)) h
      (by intro pid hp m hm;exact preprocessStep_data one c scale owners pid m x saved hm)
      (by intro m hm;exact (preprocessStep_values one c scale owners m x saved hm i).2)
      (by intro pid hp hn m;exact preprocessStep_otherDelta one c scale owners pid m (Grid.FullSchedule.legal _ work schedule pid hp) i hn)

#axiomsClean preprocess_grid
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
