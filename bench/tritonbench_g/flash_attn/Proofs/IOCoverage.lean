import bench.tritonbench_g.flash_attn.Proofs.IOValues
import bench.tritonbench_g.flash_attn.Proofs.GridValues

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000
variable {A B : Type}
variable (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
  (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
  (xs : Inputs A x)

theorem output_writes (port : Fin (outputs x).length) (i : TileIndex (outputs x)[port].shape) :
    Writes (parameters x) (programs (parameters x)) (outputs x)[port].buf
      ((memory A x).write port xs i) := by
  fin_cases port
  · let job : Program := (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val)
    let lane : TileIndex [128,x.shape.dim] :=
      (⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩,i.2.2.2.1,())
    have hr : job.1*128+lane.1.val=i.2.2.1.val := by dsimp [job,lane];omega
    have hd := flatten_decode x.shape.heads i.1.val i.2.1.val i.2.1.isLt
    refine ⟨job,output_program_present x _ _ _ ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩,
      Or.inl ⟨rfl,lane,?_,?_⟩⟩
    · simpa only [parameters,hr] using i.2.2.1.isLt
    · simp [memory,outputs,parameters,job,hr,lane,hd.1,hd.2,FlashAttn.address]
  · let job : Program := (i.2.2.1.val/128,i.1.val*x.shape.heads+i.2.1.val)
    let lane : TileIndex [128] := (⟨i.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩,())
    have hr : job.1*128+lane.1.val=i.2.2.1.val := by dsimp [job,lane];omega
    refine ⟨job,output_program_present x _ _ _ ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩,
      Or.inr ⟨rfl,lane,?_,?_⟩⟩
    · simpa only [parameters,hr] using i.2.2.1.isLt
    · simp [memory,outputs,parameters,job,hr,lane,FlashAttn.lAddress]

theorem writes_window (r : RegionName) (off : Nat)
    (h : Writes (parameters x) (programs (parameters x)) r off) :
    ∃ (port : Fin (outputs x).length) (i : TileIndex (outputs x)[port].shape),
      r=(outputs x)[port].buf ∧ off=(memory A x).write port xs i := by
  rcases h with ⟨job,hj,h⟩
  have valid := (program_valid x job).mp hj
  obtain ⟨hh,hbch,hhead⟩ := job_geometry x job valid
  rcases h with ⟨rfl,i,hi,he⟩|⟨rfl,i,hi,he⟩
  · refine ⟨(0 : Fin 2),(⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
      ⟨job.1*128+i.1.val,hi⟩,i.2.1,()),rfl,?_⟩
    simpa [memory,outputs,parameters,FlashAttn.address] using he
  · refine ⟨(1 : Fin 2),(⟨job.2/x.shape.heads,hbch⟩,⟨job.2%x.shape.heads,hhead⟩,
      ⟨job.1*128+i.1.val,hi⟩,()),rfl,?_⟩
    have merged : job.2/x.shape.heads*x.shape.heads+job.2%x.shape.heads=job.2 := by
      simpa [Nat.mul_comm] using Nat.div_add_mod job.2 x.shape.heads
    simpa [memory,outputs,parameters,FlashAttn.lAddress,merged] using he

/-- Exact equality with the previous complete-grid exterior footprint. -/
theorem outside_iff (r : RegionName) (off : Nat) :
    (sourceIO M dot encode x scale observe).Outside xs r off ↔
      ¬Writes (parameters x) (programs (parameters x)) r off := by
  constructor
  · intro outside hw
    obtain ⟨port,i,hr,ho⟩ := writes_window x xs r off hw
    rcases outside.1 port i (by trivial) with hr'|ho'
    · exact hr' hr
    · exact ho' ho
  · intro outside
    constructor
    · intro port i _
      by_cases hr : r=(outputs x)[port].buf
      · right
        intro ho
        apply outside
        rw [hr,ho]
        exact output_writes x xs port i
      · exact Or.inl hr
    · intro k
      exact k.elim0

#axiomsClean output_writes
#axiomsClean outside_iff
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
