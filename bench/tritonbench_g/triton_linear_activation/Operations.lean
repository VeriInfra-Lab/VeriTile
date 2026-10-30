import bench.tritonbench_g.triton_linear_activation.Proofs.HostTypes
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

def Request.launch (r : Request) : Prop := 0<r.batch.rows ∧ 0<r.n
instance (r : Request) : Decidable r.launch := by unfold Request.launch; infer_instance

def Allocation.kernel (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98) : Memory α :=
  if r.launch then run a r.shape h.buffers c r.mode r.activation (h.prepared a) else h.prepared a

def Allocation.copyback (h : Allocation r caller before) (a : Arithmetic α) (memory : Memory α) : Memory α :=
  match r.mode with
  | .fp32 => memory
  | .autocastFP16 => if r.hasSaved then
      copyRun a .autocastFP16 .fp32 (h.storage 4) (r.storage 3) (r.batch.rows*r.n) memory else memory

def Allocation.finalMemory (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98) : Memory α :=
  h.copyback a (h.kernel a c)

def Allocation.prepareCode (h : Allocation r caller before) : Code α Unit :=
  match r.mode with
  | .fp32 => pure ()
  | .autocastFP16 => h.copies.code .fp32 .autocastFP16

def Allocation.copybackCode (h : Allocation r caller before) : Code α Unit :=
  match r.mode with
  | .fp32 => pure ()
  | .autocastFP16 => if r.hasSaved then
      copyCode .autocastFP16 .fp32 (h.storage 4) (r.storage 3) (r.batch.rows*r.n) else pure ()

def Allocation.computeTensor (h : Allocation r caller before) (i : Fin 3) : Tensor :=
  match r.mode with
  | .fp32 => r.argument (inputSlot i)
  | .autocastFP16 => tensor (h.object (castSlot i)) (h.storage (castSlot i))
      (r.argumentShape (inputSlot i)) .autocastFP16 r.device

def Allocation.returned (h : Allocation r caller before) : Tensor :=
  tensor (match r.batch with | .matrix _ => h.object 0 | .batched _ _ => h.object 5)
    (h.storage 0) (r.batch.shape r.n) r.mode r.device

/-- The exact save_for_backward order: converted weight, optional converted
bias and converted original-rank input, never the flattened kernel input. -/
def Allocation.context (h : Allocation r caller before) : Tensor × Option Tensor × Tensor :=
  (h.computeTensor 1,if r.hasBias then some (h.computeTensor 2) else none,h.computeTensor 0)

structure HostResult (α : Type) where
  memory : Memory α
  returned : Tensor
  context : Tensor × Option Tensor × Tensor
  callerSaved : Option Tensor
  currentDevice : Nat

def Allocation.hostResult (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (currentDevice : Nat) : HostResult α :=
  ⟨h.finalMemory a c,h.returned,h.context,
    if r.hasSaved then some (r.argument 3) else none,currentDevice⟩

def Allocation.autotunedKernelCode (h : Allocation r caller before) (trials : List (Fin 98))
    (selected : Fin 98) : Code α Unit :=
  if r.launch then do
    let _ ← sequenceItems (fun c => gridCode r.shape h.buffers c r.mode r.activation) trials
    gridCode r.shape h.buffers selected r.mode r.activation
  else pure ()

def Allocation.autotunedCode (h : Allocation r caller before) (trials : List (Fin 98)) (c : Fin 98) : Code α Unit := do
  let _ ← h.prepareCode
  let _ ← h.autotunedKernelCode trials c
  h.copybackCode

end
end VeriTile.Bench.Repairs.TritonLinearActivation
