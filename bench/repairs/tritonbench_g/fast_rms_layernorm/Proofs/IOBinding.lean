import bench.repairs.tritonbench_g.fast_rms_layernorm.IO
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOBinding
open VeriTile Triton FP.Structural NativeGraph KernelTensor KernelStorage
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem row_eq {α : Type} [Inhabited α] (h : Call) (vals : Ports.Values α h) :
    rowValues h (Ports.logical h vals) "x"=Reference.row h vals := by
  funext row j
  have bound : row.val*h.cols+j.val<h.cells:=Native.offset_bound h row.val j.val row.isLt j.isLt
  simp [rowValues,Ports.logical,Reference.row,bound]

theorem upstream_eq {α : Type} [Inhabited α] (h : Call) (vals : Ports.Values α h) :
    rowValues h (Ports.logical h vals) "dx"=Reference.upstream h vals := by
  funext row j
  have bound : row.val*h.cols+j.val<h.cells:=Native.offset_bound h row.val j.val row.isLt j.isLt
  simp [rowValues,Ports.logical,Reference.upstream,bound]

theorem weight_eq {α : Type} [Inhabited α] (h : Call) (vals : Ports.Values α h) :
    weightValues h (Ports.logical h vals)=vals 1 := by
  funext j
  simp [weightValues,Ports.logical,j.isLt]

theorem saved_eq {α : Type} [Inhabited α] (h : Call) (vals : Ports.Values α h) (i : Fin h.rows) :
    Ports.logical h vals "r" i.val=vals 2 i := by simp [Ports.logical,i.isLt]

theorem real_reads (h : Call) (backward : Bool) (vals : (RealIO.program h backward).Ctx)
    (s : BlockState) (binding : (RealIO.program h backward).InputBinding s vals) :
    ∀ role,role∈KernelTensor.inputs (if backward then .backward else .forward) → ∀ off,off<cells h role →
      RealGridForward.read (IOExecution.realHeap s (role,off))=Ports.logical h vals role off := by
  change Ports.Values ℝ h at vals
  dsimp only [PreparedAlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
    RealIO.program,RealIO.memory] at binding
  intro role member off bound
  cases backward
  · simp only [Bool.false_eq_true,if_false,KernelTensor.inputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding 0 ⟨off,hb⟩ (Or.inl (by simp))
      change s.readMem "x" off=vals 0 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got
    · have hb : off<h.cols:=by simpa [cells] using bound
      have got:=binding 1 ⟨off,hb⟩ (Or.inl (by simp))
      change s.readMem "w" off=vals 1 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got
  · simp only [if_true,KernelTensor.inputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding 0 ⟨off,hb⟩ (Or.inl (by simp))
      change s.readMem "x" off=vals 0 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got
    · have hb : off<h.cols:=by simpa [cells] using bound
      have got:=binding 1 ⟨off,hb⟩ (Or.inl (by simp))
      change s.readMem "w" off=vals 1 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got
    · have hb : off<h.rows:=by simpa [cells] using bound
      have got:=binding 2 ⟨off,hb⟩ (Or.inr ⟨rfl,Or.inl (by simp)⟩)
      change s.readMem "r" off=vals 2 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding 3 ⟨off,hb⟩ (Or.inr ⟨rfl,Or.inl (by simp)⟩)
      change s.readMem "dx" off=vals 3 ⟨off,hb⟩ at got
      simpa [Ports.logical,hb] using got

theorem native_reads (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (vals : (NativeIO.program R h backward epsilon).Inputs) (s : State Term)
    (binding : (NativeIO.program R h backward epsilon).InputBinding vals s) :
    ∀ role,role∈KernelTensor.inputs (if backward then .backward else .forward) → ∀ off,off<cells h role →
      (IOExecution.nativeHeap s (role,off)).read .real=Ports.logical h (Ports.ofNative h vals) role off := by
  change ChannelValues Term (Ports.inputs h) at vals
  dsimp only [ProgramIO.InputBinding,NativeIO.program] at binding
  intro role member off bound
  cases backward
  · simp only [Bool.false_eq_true,if_false,KernelTensor.inputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding ⟨0,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inl (by simp))
      change (s.mem "x" off).read .real=vals ⟨0,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got
    · have hb : off<h.cols:=by simpa [cells] using bound
      have got:=binding ⟨1,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inl (by simp))
      change (s.mem "w" off).read .real=vals ⟨1,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got
  · simp only [if_true,KernelTensor.inputs,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding ⟨0,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inl (by simp))
      change (s.mem "x" off).read .real=vals ⟨0,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got
    · have hb : off<h.cols:=by simpa [cells] using bound
      have got:=binding ⟨1,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inl (by simp))
      change (s.mem "w" off).read .real=vals ⟨1,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got
    · have hb : off<h.rows:=by simpa [cells] using bound
      have got:=binding ⟨2,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inr ⟨rfl,Or.inl (by simp)⟩)
      change (s.mem "r" off).read .real=vals ⟨2,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got
    · have hb : off<h.cells:=by simpa [cells] using bound
      have got:=binding ⟨3,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) (Or.inr ⟨rfl,Or.inl (by simp)⟩)
      change (s.mem "dx" off).read .real=vals ⟨3,by simp [NativeIO.program,Ports.inputs]⟩ (⟨off,hb⟩,()) at got
      simpa [IOExecution.nativeHeap,Ports.logical,hb] using got

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOBinding
