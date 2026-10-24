import bench.repairs.tritonbench_g.chunk_gla_fwd.NativeIO
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInputs
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word Expr)
open NativeIO (producer consumer)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem producer_read (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (producer c I w scale).Inputs) (s : State Expr)
    (bound : (producer c I w scale).InputBinding values s)
    (r : RegionName) (off : Nat) (inside : NativeScope.inputScope c r off) :
    (s.mem r off).read .real=NativePorts.producerMemory c values r off := by
  rcases inside with ⟨role,within⟩
  rcases role with rfl|rfl|rfl
  · have h := bound (0 : Fin 3) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "q" off).read .real=NativeStaging.Expr.bits (values (0 : Fin 3) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.producerMemory,NativePorts.extend,within] using h
  · have h := bound (1 : Fin 3) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "k" off).read .real=NativeStaging.Expr.bits (values (1 : Fin 3) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.producerMemory,NativePorts.extend,within] using h
  · have h := bound (2 : Fin 3) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "g" off).read .real=NativeStaging.Expr.bits (values (2 : Fin 3) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.producerMemory,NativePorts.extend,within] using h

theorem consumer_reads (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (consumer c I w scale).Inputs) (s : State Expr)
    (bound : (consumer c I w scale).InputBinding values s) :
    NativeReads.Reads (.output c.consumer NativeRuntime.scaleTag) s (NativePorts.consumerMemory c values) := by
  intro port member off within
  simp only [NativeLaunch.inputExtents,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl
  · change off<c.heads*c.T*c.K at within
    have h := bound (0 : Fin 5) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "q" off).read .real=NativeStaging.Expr.bits (values (0 : Fin 5) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.consumerMemory,NativePorts.extend,within] using h
  · change off<c.heads*c.T*c.K at within
    have h := bound (1 : Fin 5) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "g" off).read .real=NativeStaging.Expr.bits (values (1 : Fin 5) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.consumerMemory,NativePorts.extend,within] using h
  · change off<Host.outputSize c at within
    have h := bound (2 : Fin 5) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "v" off).read .real=NativeStaging.Expr.bits (values (2 : Fin 5) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.consumerMemory,NativePorts.extend,within] using h
  · have within' : off<c.heads*c.NT*c.K*c.V := by
      simpa only [Host.Dimensions.consumer,Host.Dimensions.NT,Nat.mul_assoc] using within
    have h := bound (3 : Fin 5) (⟨off,within'⟩,⟨⟩) True.intro
    change (s.mem "h" off).read .fp32=NativeStaging.Expr.bits (values (3 : Fin 5) (⟨off,within'⟩,⟨⟩)) at h
    simpa [NativePorts.consumerMemory,NativePorts.extend,within'] using h
  · change off<Host.attentionSize c at within
    have h := bound (4 : Fin 5) (⟨off,within⟩,⟨⟩) True.intro
    change (s.mem "attention" off).read .real=NativeStaging.Expr.bits (values (4 : Fin 5) (⟨off,within⟩,⟨⟩)) at h
    simpa [NativePorts.consumerMemory,NativePorts.extend,within] using h

theorem consumer_value (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (consumer c I w scale).Inputs) (s : State Expr)
    (bound : (consumer c I w scale).InputBinding values s)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    FPReturned.value (KernelProfile.algebra w scale) NativeStaging.dot c.consumer NativeRuntime.scaleTag
      (NativeScope.consumerMemory s) h t v=
    FPReturned.value (KernelProfile.algebra w scale) NativeStaging.dot c.consumer NativeRuntime.scaleTag
      (NativePorts.consumerMemory c values) h t v := by
  let idx := OutputGrid.owner c.consumer h t v
  let lane := OutputGrid.lane c.consumer t v
  have original := NativeReads.output_value (KernelProfile.algebra w scale) NativeStaging.dot
    c.consumer NativeRuntime.scaleTag idx s (NativeScope.consumerMemory s)
    (NativeGrid.bound_reads (fun _ _=>True) NativeScope.consumerObserve _
      (NativeScope.output_ports (α:=Expr) c NativeRuntime.scaleTag).1
      (NativeScope.output_ports (α:=Expr) c NativeRuntime.scaleTag).2.1 s
      (NativeScope.consumerMemory s) (fun _ _ _=>rfl))
  have logical := NativeReads.output_value (KernelProfile.algebra w scale) NativeStaging.dot
    c.consumer NativeRuntime.scaleTag idx s (NativePorts.consumerMemory c values)
    (consumer_reads c I w scale values s bound)
  exact congrFun (original.symm.trans logical) lane

theorem flat_coordinate {heads T width : Nat} (i : Fin (heads*T*width)) :
    i.divNat.divNat.val*(T*width)+i.divNat.modNat.val*width+i.modNat.val=i.val := by
  change i.val/width/T*(T*width)+(i.val/width)%T*width+i.val%width=i.val
  rw [← Nat.mul_assoc,← Nat.add_mul,Nat.div_add_mod',Nat.div_add_mod']

#axiomsClean producer_read
#axiomsClean consumer_value
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInputs
