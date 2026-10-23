import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.RmsnormTriton.Host
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 4000000

inductive Stage where
  | admission | allocation | outputGuard | setup | execution
  deriving DecidableEq
structure Failure (α : Type) where
  stage : Stage
  memory : Memory α
  currentDevice : Nat
structure Returned (α : Type) where
  tensor : Tensor
  memory : Memory α
  currentDevice : Nat
  launchDevice : Option Nat

/-- Allocation and CUDA setup are ordinary API operations. Successful source
execution is computed below, never supplied by either operation. Device entry
and restoration follow the lexical torch.cuda.device bracket. -/
def run (x w out : TensorInfo) (eps : EpsilonInfo) (Q : Numerics α) (red : Reduction)
    (p : Parameters) (u : Nat → α) (rt : Runtime α p entry live) (callerDevice : Nat) :
    Except (Failure α) (Returned α) :=
  if checkTensorInput x w eps=.ok () then
    match rt.allocate with
    | .error _ => .error ⟨.allocation,entry,callerDevice⟩
    | .ok a =>
      if x.numel=0 then .ok ⟨p.output,a.memory,callerDevice,none⟩
      else if checkOutput x out=.ok () then
        match rt.setup a with
        | .error _ => .error ⟨.setup,a.memory,callerDevice⟩
        | .ok schedule =>
          match entryRun Q red p a u schedule with
          | none => .error ⟨.execution,a.memory,callerDevice⟩
          | some final => .ok ⟨p.output,final,callerDevice,some p.input.device⟩
      else .error ⟨.outputGuard,a.memory,callerDevice⟩
  else .error ⟨.admission,entry,callerDevice⟩

def ResultSpec (x : TensorInfo) (p : Parameters) (formula : Nat → Nat → Nat → α)
    (rt : Runtime α p entry live) (callerDevice : Nat) :
    Except (Failure α) (Returned α) → Prop
  | .error e => e.stage≠.execution ∧ e.currentDevice=callerDevice ∧
      ∀ allocation offset,live allocation → e.memory allocation offset=entry allocation offset
  | .ok result => ∃ a,rt.allocate=.ok a ∧ result.tensor=p.output ∧
      ReturnedSpec p formula a result.memory ∧ result.tensor.shape=x.shape ∧
      result.tensor.device=x.device.index ∧ result.tensor.dtype=.fp32 ∧ result.currentDevice=callerDevice ∧
      result.launchDevice=(if x.numel=0 then none else some x.device.index)

theorem complete (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry live) (api : MemoryAPI p live)
    (b : Binding x w out eps p api.byteBase) (callerDevice : Nat) :
    ResultSpec x p (logicalTensor Q red p entry) rt callerDevice
      (run x w out eps Q red p u rt callerDevice) := by
  have admitted := (checkTensorInput_iff x w eps).mpr c
  unfold run
  rw [if_pos admitted]
  cases allocated : rt.allocate with
  | error reason =>
    simp [ResultSpec]
  | ok a =>
    simp only
    have shape : p.output.shape=x.shape := a.outputShape.trans b.input.shape
    have dtype : p.output.dtype=.fp32 := a.outputDtype.trans (b.input.dtype c.input.dtype)
    have device : p.output.device=x.device.index := a.outputDevice.trans b.input.device
    by_cases empty : x.numel=0
    · rw [if_pos empty]
      refine ⟨a,allocated,rfl,?_,shape,device,dtype,rfl,by simp [empty]⟩
      refine ⟨a.outputShape,a.outputDtype,a.outputDevice,?_,fun _ _ _ => rfl,a.frame⟩
      intro batch row k hb hm hk
      have numel := b.numel x w out eps c p api.byteBase
      have positive := Nat.mul_pos (Nat.mul_pos (Nat.zero_lt_of_lt hb) (Nat.zero_lt_of_lt hm))
        (Nat.zero_lt_of_lt hk)
      omega
    · rw [if_neg empty]
      by_cases output : checkOutput x out=.ok ()
      · rw [if_pos output]
        cases setup : rt.setup a with
        | error reason =>
          simp only
          exact ⟨by simp,rfl,a.frame⟩
        | ok schedule =>
          simp only
          obtain ⟨final,ran,values,frame,entryFrame⟩ :=
            entry_result Q red p a (b.domain x w out eps c p api) u schedule
          rw [ran]
          exact ⟨a,allocated,rfl,⟨a.outputShape,a.outputDtype,a.outputDevice,values,frame,entryFrame⟩,
            shape,device,dtype,rfl,by simp [empty,b.input.device]⟩
      · rw [if_neg output]
        exact ⟨by simp,rfl,a.frame⟩

theorem rejected (x w out : TensorInfo) (eps : EpsilonInfo) (bad : ¬InputContract x w eps)
    (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry live) (callerDevice : Nat) :
    run x w out eps Q red p u rt callerDevice=.error ⟨.admission,entry,callerDevice⟩ := by
  have no : checkTensorInput x w eps≠.ok () := fun ok => bad ((checkTensorInput_iff x w eps).mp ok)
  simp [run,no]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsnormTriton.Host
