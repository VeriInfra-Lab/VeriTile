import bench.tritonbench_g.flash_attn.IO
import bench.tritonbench_g.flash_attn.Proofs.IOSafetyPrelude

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 700000
variable {A B : Type} [Inhabited A] [Inhabited B]

private def inputRegion (port : Fin 3) : RegionName :=
  if port.val=0 then "Q" else if port.val=1 then "K" else "V"

variable (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
  (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)

theorem input_value (xs : Inputs A x) (s : State B)
    (hin : (sourceIO M dot encode x scale observe).InputBinding xs s)
    (port : Fin 3) (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    (s.mem (inputRegion port)
      (FlashAttn.address (inputStrides x port) i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real =
      encode (array xs port i) := by
  fin_cases port
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides,array,mapValue] using
      hin (0 : Fin 3) i (by trivial)
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides,array,mapValue] using
      hin (1 : Fin 3) i (by trivial)
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides,array,mapValue] using
      hin (2 : Fin 3) i (by trivial)

theorem input_bound (xs : Inputs A x) (bounds : RegionName → Nat)
    (hb : (sourceIO M dot encode x scale observe).InBounds bounds xs)
    (port : Fin 3) (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    FlashAttn.address (inputStrides x port) i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val <
      bounds (inputRegion port) := by
  fin_cases port
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides] using
      hb.2.1 (0 : Fin 3) i (by trivial)
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides] using
      hb.2.1 (1 : Fin 3) i (by trivial)
  · simpa [sourceIO,memory,inputs,inputRegion,inputStrides] using
      hb.2.1 (2 : Fin 3) i (by trivial)

theorem job_geometry (job : Program) (valid : job.Valid x) :
    0<x.shape.heads ∧ job.2/x.shape.heads<x.shape.batches ∧ job.2%x.shape.heads<x.shape.heads := by
  have hh : 0<x.shape.heads := by
    by_contra h
    have hz : x.shape.heads=0 := Nat.eq_zero_of_not_pos h
    have impossible := valid.2
    simp [hz] at impossible
  exact ⟨hh,(Nat.div_lt_iff_lt_mul hh).mpr valid.2,Nat.mod_lt _ hh⟩

/-- Every actual source address is one of the exposed logical channel windows.
The proof does not require positive dimensions or injective output strides. -/
theorem source_bounds (xs : Inputs A x) (bounds : RegionName → Nat)
    (hb : (sourceIO M dot encode x scale observe).InBounds bounds xs)
    (job : Program) (valid : job.Valid x) (mem : Mem B) :
    IOSourceSafety.Bounds (parameters x) (stateAt (parameters x) job mem) bounds := by
  obtain ⟨hh,hbch,hhead⟩ := job_geometry x job valid
  constructor
  · intro i hi
    have h := input_bound M dot encode x scale observe xs bounds hb 0
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨job.1*128+i.1.val,hi⟩,i.2.1,())
    simpa [parameters,base,stateAt,FlashAttn.address,inputRegion,inputStrides] using h
  · intro row i hi
    have h := input_bound M dot encode x scale observe xs bounds hb 1
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨row+i.2.1.val,hi⟩,i.1,())
    simpa [parameters,base,stateAt,FlashAttn.address,inputRegion,inputStrides,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro row i hi
    have h := input_bound M dot encode x scale observe xs bounds hb 2
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨row+i.1.val,hi⟩,i.2.1,())
    simpa [parameters,base,stateAt,FlashAttn.address,inputRegion,inputStrides] using h
  · intro i hi
    have h := hb.2.2.1 (0 : Fin 2)
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨job.1*128+i.1.val,hi⟩,i.2.1,()) (by trivial)
    simpa [sourceIO,memory,outputs,parameters,base,stateAt,ActualSource.address,FlashAttn.address] using h
  · intro i hi
    have h := hb.2.2.1 (1 : Fin 2)
      (⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
       ⟨job.1*128+i.1.val,hi⟩,()) (by trivial)
    have merged : job.2/x.shape.heads*x.shape.heads+job.2%x.shape.heads=job.2 := by
      simpa [Nat.mul_comm] using Nat.div_add_mod job.2 x.shape.heads
    simpa [sourceIO,memory,outputs,parameters,stateAt,ActualSource.lAddress,FlashAttn.lAddress,merged] using h

#axiomsClean source_bounds
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
