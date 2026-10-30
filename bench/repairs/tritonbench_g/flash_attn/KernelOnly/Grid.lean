import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Output

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

abbrev Program := Nat × Nat
abbrev Mem (α : Type) := RegionName→Nat→Cell α

def stateAt (p : Parameters) (job : Program) (mem : Mem α) : State α where
  mem:=mem
  regs:=fun _ _ _=>none
  pids:=fun ax=>if ax=0 then job.1 else if ax=1 then job.2 else 0
  numPids:=fun ax=>if ax=0 then (p.SEQLEN+127)/128 else if ax=1 then p.BS*p.HEAD else 1
  undef:=fun d _ _=>defaultValue d

def programs (p : Parameters) : List Program :=
  (List.range ((p.SEQLEN+127)/128)).flatMap (fun row=>
    (List.range (p.BS*p.HEAD)).map (fun head=>(row,head)))

def runProgram (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (mem : Mem α) : Option (Mem α) :=
  (FP.Structural.exec (engine (withDot M dot)) (kernel p) (bindScale scale (stateAt p job mem))).map State.mem

def referenceProgram (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (mem : Mem α) : Mem α :=
  (written M p (stateAt p job mem) (accumulated M dot p (stateAt p job mem) scale)).mem

def runGrid (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α) :
    List Program→Mem α→Option (Mem α)
  | [],mem=>some mem
  | job::rest,mem=>do runGrid M dot p scale rest (←runProgram M dot p scale job mem)

def referenceGrid (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α) :
    List Program→Mem α→Mem α
  | [],mem=>mem
  | job::rest,mem=>referenceGrid M dot p scale rest (referenceProgram M dot p scale job mem)

theorem run_program (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (mem : Mem α) :
    runProgram M dot p scale job mem=some (referenceProgram M dot p scale job mem) := by
  obtain ⟨final,run,memory⟩:=source_memory M dot p (stateAt p job mem) scale
  simp only [runProgram,run,Option.map_some,memory,referenceProgram]

theorem source_grid (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (jobs : List Program) (mem : Mem α) :
    runGrid M dot p scale jobs mem=some (referenceGrid M dot p scale jobs mem) := by
  induction jobs generalizing mem with
  | nil=>rfl
  | cons job rest ih=>simpa only [runGrid,run_program,Option.bind_some,referenceGrid] using ih _

def Writes (p : Parameters) (jobs : List Program) (r : RegionName) (off : Nat) : Prop :=
  ∃ job∈jobs,
    (r="O" ∧ ∃ i : TileIndex [128,p.DIM],job.1*128+i.1.val<p.SEQLEN ∧
      off=job.2/p.HEAD*p.stride_o_bs+job.2%p.HEAD*p.stride_o_head+
        (job.1*128+i.1.val)*p.stride_o_seqlen+i.2.1.val*p.stride_o_dim) ∨
    (r="L" ∧ ∃ i : TileIndex [128],job.1*128+i.1.val<p.SEQLEN ∧
      off=job.2*p.SEQLEN+(job.1*128+i.1.val))

theorem program_frame (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (mem : Mem α) (r : RegionName) (off : Nat)
    (outside : ¬Writes p [job] r off) : referenceProgram M dot p scale job mem r off=mem r off := by
  have hout : r≠"O" ∨ ∀ i : TileIndex [128,p.DIM],
      active p (stateAt p job mem) i → off≠address p (stateAt p job mem) i := by
    by_cases hr : r="O"
    · right;intro i hi he
      apply outside
      refine ⟨job,by simp,Or.inl ⟨hr,i,?_,?_⟩⟩
      · simpa [active,stateAt] using hi
      · simpa [address,base,stateAt] using he
    · exact Or.inl hr
  have hl : r≠"L" ∨ ∀ i : TileIndex [128],
      lActive p (stateAt p job mem) i → off≠lAddress p (stateAt p job mem) i := by
    by_cases hr : r="L"
    · right;intro i hi he
      apply outside
      refine ⟨job,by simp,Or.inr ⟨hr,i,?_,?_⟩⟩
      · simpa [lActive,stateAt] using hi
      · simpa [lAddress,stateAt] using he
    · exact Or.inl hr
  unfold referenceProgram written
  rw [State.masked_scatter_frame]
  · rw [State.masked_scatter_frame]
    · rfl
    · rcases hl with hr|ho
      · exact Or.inl hr
      · exact Or.inr (fun i _ hi=>ho i hi)
  · rcases hout with hr|ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi=>ho i hi)

theorem grid_frame (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (jobs : List Program) (mem : Mem α) (r : RegionName) (off : Nat)
    (outside : ¬Writes p jobs r off) : referenceGrid M dot p scale jobs mem r off=mem r off := by
  induction jobs generalizing mem with
  | nil=>rfl
  | cons job rest ih=>
    have restOut : ¬Writes p rest r off := by
      rintro ⟨j,hj,h⟩;exact outside ⟨j,List.mem_cons_of_mem job hj,h⟩
    have jobOut : ¬Writes p [job] r off := by
      rintro ⟨j,hj,h⟩;simp only [List.mem_singleton] at hj;subst j
      exact outside ⟨job,by simp,h⟩
    rw [referenceGrid,ih _ restOut,program_frame M dot p scale job mem r off jobOut]

theorem input_frame (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (jobs : List Program) (mem : Mem α) (r : RegionName) (hr : r∈(["Q","K","V"] : List RegionName)) (off : Nat) :
    referenceGrid M dot p scale jobs mem r off=mem r off := by
  apply grid_frame
  rintro ⟨job,_,h⟩
  rcases h with ⟨rfl,_⟩|⟨rfl,_⟩ <;> simp at hr

#axiomsClean source_grid
#axiomsClean grid_frame
#axiomsClean input_frame
#specNonCircular referenceGrid avoiding [run,FP.Structural.exec,runProgram,runGrid]
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
