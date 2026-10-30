import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Memory

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural
noncomputable section
open scoped Classical
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

theorem grid_cell (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (r : RegionName) (off : Nat) (target : Program) (cell : Cell α) (baseMemory : Mem α)
    (value : ∀ memory,InputsEq memory baseMemory → referenceProgram M dot p scale target memory r off=cell)
    (jobs : List Program) (present : target∈jobs)
    (unique : ∀ job∈jobs,Writes p [job] r off→job=target)
    (memory : Mem α) (inputs : InputsEq memory baseMemory) :
    referenceGrid M dot p scale jobs memory r off=cell := by
  induction jobs generalizing memory with
  | nil=>simp at present
  | cons job rest ih=>
    by_cases later : target∈rest
    · have source : InputsEq (referenceProgram M dot p scale job memory) memory := by
        exact grid_inputs M dot p scale [job] memory
      exact ih later (fun j hj=>unique j (List.mem_cons_of_mem job hj)) _
        (fun r hr off=>(source r hr off).trans (inputs r hr off))
    · have he : target=job := by simpa [later] using present
      subst job
      have outside : ¬Writes p rest r off := by
        rintro ⟨j,hj,hwrite⟩
        have equal:=unique j (List.mem_cons_of_mem target hj) ⟨j,by simp,hwrite⟩
        exact later (equal ▸ hj)
      rw [referenceGrid,grid_frame M dot p scale rest _ r off outside]
      exact value memory inputs

theorem output_writer_unique (x : Layout) (hs : x.shape.Valid) (injective : OutputInjective x)
    (b h row col : Nat) (active : (Work.output b h row col).active x.shape)
    (job : Program) (valid : job.Valid x)
    (write : Writes (parameters x) [job] "O" (FlashAttn.address x.o b h row col)) :
    job=(row/128,b*x.shape.heads+h) := by
  rcases write with ⟨j,hj,hw⟩
  simp only [List.mem_singleton] at hj
  subst j
  rcases hw with ⟨_,i,hi,he⟩|⟨hr,_⟩
  · have hb : job.2/x.shape.heads<x.shape.batches := (Nat.div_lt_iff_lt_mul hs.2.1).mpr valid.2
    have hh : job.2%x.shape.heads<x.shape.heads := Nat.mod_lt _ hs.2.1
    have same:=injective b h row col (job.2/x.shape.heads) (job.2%x.shape.heads)
      (job.1*128+i.1.val) i.2.1.val active ⟨hb,hh,hi,i.2.1.isLt⟩
      (by simpa [parameters,FlashAttn.address] using he)
    apply Prod.ext
    · dsimp
      have lane:=i.1.isLt
      omega
    · dsimp
      rw [same.1,same.2.1]
      simpa [Nat.mul_comm] using (Nat.div_add_mod job.2 x.shape.heads).symm
  · simp at hr

theorem log_writer_unique (x : Layout) (b h row : Nat)
    (active : (Work.logsum b h row).active x.shape) (job : Program)
    (write : Writes (parameters x) [job] "L" (FlashAttn.lAddress x.shape b h row)) :
    job=(row/128,b*x.shape.heads+h) := by
  rcases write with ⟨j,hj,hw⟩
  simp only [List.mem_singleton] at hj
  subst j
  rcases hw with ⟨hr,_⟩|⟨_,i,hi,he⟩
  · simp at hr
  · have lhs:=flatten_decode x.shape.length (b*x.shape.heads+h) row active.2.2
    have rhs:=flatten_decode x.shape.length job.2 (job.1*128+i.1.val) hi
    change (b*x.shape.heads+h)*x.shape.length+row=job.2*x.shape.length+(job.1*128+i.1.val) at he
    have hd:=congrArg (fun z=>z/x.shape.length) he
    have hm:=congrArg (fun z=>z%x.shape.length) he
    simp only [lhs.1,rhs.1] at hd
    simp only [lhs.2,rhs.2] at hm
    apply Prod.ext
    · dsimp
      have lane:=i.1.isLt
      omega
    · exact hd.symm

theorem output_program_present (x : Layout) (b h row : Nat)
    (active : (Work.logsum b h row).active x.shape) :
    (row/128,b*x.shape.heads+h)∈programs (parameters x) := by
  apply (program_valid x _).mpr
  constructor
  · have := active.2.2;omega
  · have := active.1
    have := active.2.1
    nlinarith

#axiomsClean grid_cell
#axiomsClean output_writer_unique
#axiomsClean log_writer_unique
#axiomsClean output_program_present
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
