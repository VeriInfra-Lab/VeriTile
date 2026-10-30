import bench.tritonbench_g.flash_attn.Proofs.IOBindings
import bench.tritonbench_g.flash_attn.Targets

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000
variable {A B : Type} [Inhabited A] [Inhabited B]
theorem checkRow (r : RegionName) (base N D m rs ds start : Nat) (i : TileIndex [m,D]) :
    (BlockPtr.mk r base [N,D] [m,D] [rs,ds] [start,0]).inBounds
      (TileShape.indexToList _ i) [0]=decide (start+i.1.val<N) := by
  change (decide (start+i.1.val<N) && Bool.true)=_
  simp

theorem checkColumn (r : RegionName) (base N D n ds rs start : Nat) (i : TileIndex [D,n]) :
    (BlockPtr.mk r base [D,N] [D,n] [ds,rs] [0,start]).inBounds
      (TileShape.indexToList _ i) [1]=decide (start+i.2.1.val<N) := by
  change (decide (start+i.2.1.val<N) && Bool.true)=_
  simp

variable (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
  (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
  (xs : Inputs A x) (s : State B)
  (hin : (sourceIO M dot encode x scale observe).InputBinding xs s)
  (job : Program) (valid : job.Valid x)

include hin valid

theorem query_binding :
    ActualSource.query M (parameters x) (stateAt (parameters x) job s.mem) scale =
      Logical.query M x.shape.dim x.shape.length job.1
        (fun r d => encode (inputAt xs 0 (job.2/x.shape.heads) (job.2%x.shape.heads) r d)) scale := by
  obtain ⟨hh,hbch,hhead⟩ := job_geometry x job valid
  funext i
  by_cases hi : job.1*128+i.1.val<x.shape.length
  · have h := input_value M dot encode x scale observe xs s hin 0
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨job.1*128+i.1.val,hi⟩,i.2.1,())
    have read : (s.mem "Q" (FlashAttn.address x.q (job.2/x.shape.heads) (job.2%x.shape.heads)
        (job.1*128+i.1.val) i.2.1.val)).read .real =
        encode (inputAt xs 0 (job.2/x.shape.heads) (job.2%x.shape.heads) (job.1*128+i.1.val) i.2.1.val) := by
      simpa [inputStrides,inputAt,hbch,hhead,hi,(show i.2.1.val<x.shape.dim from i.2.1.isLt)] using h
    simpa [ActualSource.query,Logical.query,qptr,check0,parameters,base,stateAt,zero,
      hi,FlashAttn.address] using congrArg (fun a => M.cast (some .fp32) .real .fp16
        (M.binary (some .fp32) .real .mul (M.fp32Load a)
          (M.binary (some .fp32) .real .mul scale (M.literal (some .fp32) .real 1.44269504)))) read
  · simp [ActualSource.query,Logical.query,qptr,check0,parameters,base,stateAt,zero,hi]

theorem key_binding (row : Nat) :
    ActualSource.key M (parameters x) (stateAt (parameters x) job s.mem) row =
      Logical.key M x.shape.dim x.shape.length row
        (fun r d => encode (inputAt xs 1 (job.2/x.shape.heads) (job.2%x.shape.heads) r d)) := by
  obtain ⟨hh,hbch,hhead⟩ := job_geometry x job valid
  funext i
  by_cases hi : row+i.2.1.val<x.shape.length
  · have h := input_value M dot encode x scale observe xs s hin 1
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,⟨row+i.2.1.val,hi⟩,i.1,())
    have read : (s.mem "K" (FlashAttn.address x.k (job.2/x.shape.heads) (job.2%x.shape.heads)
        (row+i.2.1.val) i.1.val)).read .real =
        encode (inputAt xs 1 (job.2/x.shape.heads) (job.2%x.shape.heads) (row+i.2.1.val) i.1.val) := by
      simpa [inputStrides,inputAt,hbch,hhead,hi,(show i.1.val<x.shape.dim from i.1.isLt)] using h
    simpa [ActualSource.key,Logical.key,kptr,checkColumn,parameters,base,stateAt,zero,hi,FlashAttn.address,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using congrArg M.fp32Load read
  · simp [ActualSource.key,Logical.key,kptr,checkColumn,parameters,base,stateAt,zero,hi]

theorem value_binding (row : Nat) :
    ActualSource.value M (parameters x) (stateAt (parameters x) job s.mem) row =
      Logical.value M x.shape.dim x.shape.length row
        (fun r d => encode (inputAt xs 2 (job.2/x.shape.heads) (job.2%x.shape.heads) r d)) := by
  obtain ⟨hh,hbch,hhead⟩ := job_geometry x job valid
  funext i
  by_cases hi : row+i.1.val<x.shape.length
  · have h := input_value M dot encode x scale observe xs s hin 2
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,⟨row+i.1.val,hi⟩,i.2.1,())
    have read : (s.mem "V" (FlashAttn.address x.v (job.2/x.shape.heads) (job.2%x.shape.heads)
        (row+i.1.val) i.2.1.val)).read .real =
        encode (inputAt xs 2 (job.2/x.shape.heads) (job.2%x.shape.heads) (row+i.1.val) i.2.1.val) := by
      simpa [inputStrides,inputAt,hbch,hhead,hi,(show i.2.1.val<x.shape.dim from i.2.1.isLt)] using h
    simpa [ActualSource.value,Logical.value,vptr,checkRow,parameters,base,stateAt,zero,hi,FlashAttn.address]
      using congrArg M.fp32Load read
  · simp [ActualSource.value,Logical.value,vptr,checkRow,parameters,base,stateAt,zero,hi]

theorem accumulated_binding :
    ActualSource.accumulated M dot (parameters x) (stateAt (parameters x) job s.mem) scale =
      Logical.rowAccumulator M dot encode x scale xs job := by
  have q := query_binding M dot encode x scale observe xs s hin job valid
  have k := key_binding M dot encode x scale observe xs s hin job valid
  have v := value_binding M dot encode x scale observe xs s hin job valid
  have update_eq : ActualSource.update M dot (parameters x) (stateAt (parameters x) job s.mem) scale =
      Logical.update M dot (parameters x) job.1
        (fun r d => encode (inputAt xs 0 (job.2/x.shape.heads) (job.2%x.shape.heads) r d))
        (fun r d => encode (inputAt xs 1 (job.2/x.shape.heads) (job.2%x.shape.heads) r d))
        (fun r d => encode (inputAt xs 2 (job.2/x.shape.heads) (job.2%x.shape.heads) r d)) scale := by
    funext row a
    delta ActualSource.update ActualSource.maximum ActualSource.probability ActualSource.rescale
      ActualSource.scores ActualSource.contribution
    rw [q,k row,v row]
    rfl
  unfold ActualSource.accumulated Logical.rowAccumulator Logical.accumulated
  rw [update_eq]
  rfl

#axiomsClean accumulated_binding
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
