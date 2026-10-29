import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeInputs
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelNative

/-! The finite public native channels observe exactly the original physical
project/flush endpoints. Input allocations may alias each other; the existing
physical and Tensor theorems retain their original writable-freshness premises.
No new source execution, retagging step or numerical law is assumed here. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativePhysical
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word Expr)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

def producerValues (abi : Physical.ABI c) (before : Nat → Word) :
    ChannelValues Word (NativePorts.producerInputs c) :=
  Fin.cases (fun i=>before ((abi.alloc "q").base+i.1.val))
    (Fin.cases (fun i=>before ((abi.alloc "k").base+i.1.val))
      (Fin.cases (fun i=>before ((abi.alloc "g").base+i.1.val)) (fun i=>Fin.elim0 i)))

def consumerValues (abi : Physical.ABI c) (before : Nat → Word) :
    ChannelValues Word (NativePorts.consumerInputs c) :=
  Fin.cases (fun i=>before ((abi.alloc "q").base+i.1.val))
    (Fin.cases (fun i=>before ((abi.alloc "g").base+i.1.val))
      (Fin.cases (fun i=>before ((abi.alloc "v").base+i.1.val))
        (Fin.cases (fun i=>before ((abi.alloc "h").base+i.1.val))
          (Fin.cases (fun i=>before ((abi.alloc "attention").base+i.1.val)) (fun i=>Fin.elim0 i)))))

def initial (abi : Physical.ABI c) (before : Nat → Word) : State Expr :=
  FPStorage.state (FPStorage.project abi.alloc (NativeSource.inputMemory before)) (fun _=>0) (fun _=>0)

theorem producer_binding (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (abi : Physical.ABI c) (before : Nat → Word) :
    (NativeIO.producer c I w scale).InputBinding (producerValues abi before) (initial abi before) := by
  intro channel i _
  fin_cases channel
  · exact FPStorage.import_real _ "q" i.1.val (by decide)
  · exact FPStorage.import_real _ "k" i.1.val (by decide)
  · exact FPStorage.import_real _ "g" i.1.val (by decide)

theorem consumer_binding (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (abi : Physical.ABI c) (before : Nat → Word) :
    (NativeIO.consumer c I w scale).InputBinding (consumerValues abi before) (initial abi before) := by
  intro channel i _
  fin_cases channel
  · exact FPStorage.import_real _ "q" i.1.val (by decide)
  · exact FPStorage.import_real _ "g" i.1.val (by decide)
  · exact FPStorage.import_real _ "v" i.1.val (by decide)
  · exact FPStorage.import_history _ i.1.val
  · exact FPStorage.import_real _ "attention" i.1.val (by decide)

theorem consumer_memory {α : Type} [Inhabited α] (memory : FPStorage.Memory α)
    (pids numPids : Nat → Nat) :
    NativeScope.consumerMemory (FPStorage.state memory pids numPids)=memory := by
  funext r off
  by_cases output : r="o"
  · subst r
    simp [NativeScope.consumerMemory,NativeScope.consumerObserve,FPStorage.state,
      FPStorage.importMemory]
  · simp only [NativeScope.consumerMemory,NativeScope.consumerObserve,if_neg output]
    by_cases history : r="h"
    · subst r
      exact FPStorage.import_history memory off
    · rw [if_neg history]
      exact FPStorage.import_real memory r off history

/-- Both independent native targets agree on every finite output with the
same full-source physical project/flush execution used by KernelNative. -/
theorem producer_target (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (abi : Physical.ABI c) (before : Nat → Word) (off : Nat) (inside : off<Host.attentionSize c) :
    NativePorts.attentionWord c I w scale (producerValues abi before) off=
      (KernelRuntime.reference .producer w scale abi before ((abi.alloc "attention").base+off)).eval I := by
  obtain ⟨logical,logicalRun,logicalValues,_⟩ := NativeWhole.producer
    (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag (initial abi before)
    (NativePorts.producerMemory c (producerValues abi before))
    (NativeInputs.producer_read c I w scale _ _ (producer_binding I w scale abi before))
  have physicalInput (r : RegionName) (i : Nat) (input : NativeScope.inputScope c r i) :
      ((initial abi before).mem r i).read .real=
        FPStorage.project abi.alloc (NativeSource.inputMemory before) r i := by
    apply FPStorage.import_real
    rcases input.1 with rfl|rfl|rfl <;> decide
  obtain ⟨physical,physicalRun,physicalValues,_⟩ := NativeWhole.producer
    (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag (initial abi before)
    (FPStorage.project abi.alloc (NativeSource.inputMemory before)) physicalInput
  have same : logical=physical := Option.some.inj (logicalRun.symm.trans physicalRun)
  subst physical
  have agreement := (logicalValues off inside).symm.trans (physicalValues off inside)
  change _=(NativeStaging.lower (Physical.producerReference (KernelProfile.algebra w scale)
    NativeStaging.dot abi NativeRuntime.scaleTag (NativeSource.inputMemory before)
      ((abi.alloc "attention").base+off))).eval I
  rw [Physical.producer_readback _ _ _ _ _ off inside]
  exact congrArg (fun expression=>(NativeStaging.lower expression).eval I) agreement

theorem consumer_target (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (abi : Physical.ABI c) (before : Nat → Word) (i : Fin (Host.outputSize c)) :
    NativePorts.outputWord c I w scale (consumerValues abi before) i=
      (KernelRuntime.reference .consumer w scale abi before ((abi.alloc "o").base+i.val)).eval I := by
  have agreement := NativeInputs.consumer_value c I w scale (consumerValues abi before) (initial abi before)
    (consumer_binding I w scale abi before) i.divNat.divNat i.divNat.modNat i.modNat
  dsimp only [initial] at agreement
  rw [consumer_memory] at agreement
  have returned := FPReturned.consumer_readback (KernelProfile.algebra w scale) NativeStaging.dot c
    NativeRuntime.scaleTag (FPStorage.project abi.alloc (NativeSource.inputMemory before))
      i.divNat.divNat i.divNat.modNat i.modNat
  rw [NativeInputs.flat_coordinate] at returned
  change _=(NativeStaging.lower (Physical.consumerReference (KernelProfile.algebra w scale)
    NativeStaging.dot abi NativeRuntime.scaleTag (NativeSource.inputMemory before)
      ((abi.alloc "o").base+i.val))).eval I
  rw [Physical.consumer_readback _ _ _ _ _ i.val i.isLt]
  exact congrArg (fun expression=>(NativeStaging.lower expression).eval I) (agreement.symm.trans returned.symm)

#axiomsClean producer_binding
#axiomsClean consumer_binding
#axiomsClean producer_target
#axiomsClean consumer_target
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativePhysical
