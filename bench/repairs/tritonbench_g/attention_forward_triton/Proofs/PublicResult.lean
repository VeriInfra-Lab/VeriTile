import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.PublicSafety
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

def realState (p : Parameters) (vals : Inputs Real.R p) (s : State Real.R) : State Real.R :=
  {s with mem := fun r off => .mk .real (logicalRead (some 0) p vals r off)}

def realValues (p : Parameters) (vals : Inputs Real.R p) (r : RegionName) (off : Nat) : ℝ :=
  (logicalRead (some 0) p vals r off).getD 0

theorem real_read_finite (p : Parameters) (vals : Inputs Real.R p) (finite : RealInputs p vals)
    (r : RegionName) (off : Nat) :
    some (realValues p vals r off)=logicalRead (some 0) p vals r off := by
  have h : (logicalRead (some 0) p vals r off).isSome := by
    unfold logicalRead
    split
    · exact finite _ _
    · trivial
  unfold realValues
  generalize logicalRead (some 0) p vals r off=value at h ⊢
  cases value <;> simp_all

theorem real_state_reads (p : Parameters) (vals : Inputs Real.R p) (finite : RealInputs p vals)
    (s : State Real.R) : Real.Reads p (realState p vals s) (realValues p vals) := by
  intro r _ off
  change Cell.mk .real (logicalRead (some 0) p vals r off)=_
  rw [←real_read_finite p vals finite r off]
  rfl

theorem real_formula (p : Parameters) (work : List Program) (vals : Inputs Real.R p)
    (finite : RealInputs p vals) (s : State Real.R)
    (bound : (realIO p work).InputBinding vals s) (j : DataLane p) :
    answer Real.algebra Real.dotModel p
      (start (grid p) (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0) s)
      (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt)=
      mathematical p vals ⟨0,by simp [outputChannels]⟩ j := by
  let pid : Program := (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0)
  have legal : pid∈programs (grid p) :=
    (grid_covers p _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1
  have reads : Logical.Reads p (start (grid p) pid s) (start (grid p) pid (realState p vals s)) := by
    apply reads_of_occurrences p s (realState p vals s) pid legal
    intro r off occurs
    exact read_extension (some 0) id p vals s (real_read_binding p work vals s bound) r off occurs
  rw [Logical.answer_eq Real.algebra Real.dotModel p _ _ reads]
  have active : Source.active p (start (grid p) pid (realState p vals s))
      (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt) := by
    constructor
    · change j.2.2.1.val/128*128+j.2.2.1.val%128<p.length
      have := j.2.2.1.isLt
      omega
    · exact j.2.2.2.1.isLt
  have modelReads : Real.Reads p (start (grid p) pid (realState p vals s)) (realValues p vals) :=
    real_state_reads p vals finite s
  rw [Real.answer_real p _ (realValues p vals) modelReads _ active,
    Real.expected_at p (realState p vals s) (realValues p vals) _ _ _ _ j.2.1.isLt j.2.2.2.1.isLt]
  rfl

def wordState (p : Parameters) (vals : Inputs Native.Word p) : State Native.Word where
  mem := fun r off => .mk .real (logicalRead 0 p vals r off)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem native_formula (I : Native.Instructions) (p : Parameters) (work : List Program)
    (vals : Inputs Native.Word p) (s : State Native.Expr)
    (bound : (nativeIO I p work).InputBinding vals s) (j : DataLane p) :
    (Native.lower (answer Native.algebra Native.dotModel p
      (start (grid p) (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0) s)
      (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt)
      (Native.physicalOwner j.2.2.2.1.val))).eval I.ops=
      nativeTarget I p vals ⟨0,by simp [outputChannels]⟩ j := by
  let pid : Program := (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0)
  have legal : pid∈programs (grid p) :=
    (grid_covers p _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1
  have reads : Logical.Reads p (start (grid p) pid s)
      (start (grid p) pid (Native.injectState (wordState p vals))) := by
    apply reads_of_occurrences p s (Native.injectState (wordState p vals)) pid legal
    intro r off occurs
    exact read_extension 0 wordEncode p vals s (native_read_binding I p work vals s bound) r off occurs
  rw [Logical.answer_eq Native.algebra Native.dotModel p _ _ reads]
  change Native.outputWord I p (wordState p vals) j.1.val j.2.1.val
    j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt=_
  rw [NativeReference.target_binding I p (wordState p vals) _ _ _ _
    j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt]
  rfl

theorem exterior (p : Parameters) (r : RegionName) (off : Nat)
    (outside : ∀ j : DataLane p, r≠p.Out ∨ off≠dataAddress p.o j) :
    r≠p.Out ∨ ¬Footprint p off := by
  by_cases hr : r=p.Out
  · right
    rintro ⟨b,hb,h,hh,n,hn,d,hd,equal⟩
    exact (outside (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,())).resolve_left
      (fun ne => ne hr) equal
  · exact Or.inl hr

theorem real_implements (p : Parameters) (sep : Separated p)
    (injective : RawKernel.OutputInjective p) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    (realIO p work).ImplementsOn (RealInputs p) (mathematical p) := by
  intro vals finite bounds backed s bound
  have input : ∀ i j,inputAddress p i j<bounds (inputChannels p)[i].buf :=
    fun i j => backed.2.1 i j trivial
  have output : ∀ j : DataLane p,dataAddress p.o j<bounds p.Out := by
    intro j
    simpa [realIO,memory,outputChannels,Fin.cases] using
      backed.2.2.1 ⟨0,by simp [realIO,memory,outputChannels]⟩ j trivial
  obtain ⟨t,ran,values,frame⟩ := RawKernel.grid_result (fun _ => Real.algebra)
    (fun _ => Real.dotModel) p sep injective s work legal
  refine ⟨schedule_safe Real.algebra Real.dotModel p work
    (fun pid member => legal.mem_iff.mpr member) bounds input output s,t,?_,?_,?_⟩
  · exact (schedule_run Real.algebra Real.dotModel p work s).trans ran
  · intro o j _
    fin_cases o
    change mapCell id (t.mem p.Out (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=
      Cell.mk .fp16 (mathematical p vals ⟨0,by simp [outputChannels]⟩ j)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt,
      real_formula p work vals finite s bound j]
    rfl
  · intro r off outside
    apply frame r off
    apply exterior p r off
    intro j
    simpa [realIO,memory,outputChannels,Fin.cases] using
      outside.1 ⟨0,by simp [realIO,memory,outputChannels]⟩ j trivial

theorem native_implements (I : Native.Instructions) (p : Parameters) (sep : Separated p)
    (injective : RawKernel.OutputInjective p) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    (nativeIO I p work).Implements (nativeTarget I p) := by
  intro vals _ bounds backed s bound
  have input : ∀ i j,inputAddress p i j<bounds (inputChannels p)[i].buf :=
    fun i j => backed.2.1 i j trivial
  have output : ∀ j : DataLane p,dataAddress p.o j<bounds p.Out := by
    intro j
    simpa [nativeIO,memory,outputChannels,Fin.cases] using
      backed.2.2.1 ⟨0,by simp [nativeIO,memory,outputChannels]⟩ j trivial
  obtain ⟨t,ran,values,frame⟩ := RawKernel.grid_result (fun _ => Native.algebra)
    (fun _ => Native.dotModel) p sep injective s work legal
  refine ⟨schedule_safe Native.algebra Native.dotModel p work
    (fun pid member => legal.mem_iff.mpr member) bounds input output s,t,?_,?_,?_⟩
  · exact (schedule_run Native.algebra Native.dotModel p work s).trans ran
  · intro o j _
    fin_cases o
    change mapCell (fun value => (Native.lower (value (Native.physicalOwner j.2.2.2.1.val))).eval I.ops)
      (t.mem p.Out (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=Cell.mk .fp16 (nativeTarget I p vals ⟨0,by simp [outputChannels]⟩ j)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt]
    exact congrArg (Cell.mk .fp16) (native_formula I p work vals s bound j)
  · intro r off outside
    apply frame r off
    apply exterior p r off
    intro j
    simpa [nativeIO,memory,outputChannels,Fin.cases] using
      outside.1 ⟨0,by simp [nativeIO,memory,outputChannels]⟩ j trivial

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
