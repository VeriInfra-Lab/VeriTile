import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PairIO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.Pair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open Core
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 40000
variable {α : Type} [Inhabited α]

def Exterior (p : Parameters) (r : RegionName) (o : Nat) : Prop :=
  r≠"C" ∨ ∀ x : LogicalIndex p,o≠logicalAddress p x
def Frame (p : Parameters) (s t : State α) : Prop :=
  ∀ r o,Exterior p r o → t.mem r o=s.mem r o

/-- The exact scatter footprint does not require an arithmetic output formula,
nor the zero-conversion identity used by the optional native result. -/
theorem suffix_frame (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (tile : Nat)
    (inv : InnerInvariant P p initial tile p.problem.kTiles current) :
    Frame p initial (suffixResult P p initial current tile) := by
  intro r o outside
  rw [suffixMemory P p initial current tile inv]
  unfold written
  apply State.masked_scatter_frame
  rcases outside with region | offset
  · exact Or.inl region
  · right
    intro i _ active
    rw [address_index p tile i active]
    exact offset (index p tile i active)

theorem tile_frame (P : Source.Arithmetic α α) (p : Parameters)
    (initial : State α) (pid ordinal : Nat) (regs : Registers p pid initial)
    (tileBound : p.problem.tile pid ordinal<p.problem.tiles) :
    ∃ final,run (backend P) (outer p)
        (initial.setReg "tile_ordinal" .int [] (fun _ => (ordinal : Int)))=some final ∧
      Registers p pid final ∧ Frame p initial final := by
  let setup := tilePreludeExecution P p pid ordinal initial regs
  have invariant := tile_initial P p pid ordinal initial regs tileBound
  have registers := tile_registers P p pid ordinal initial regs
  obtain ⟨last,ran,inv,reg⟩ := innerLoop P p initial setup.val pid (p.problem.tile pid ordinal) 0
    (Nat.zero_le _) invariant registers
  simp only [Int.natCast_zero] at ran
  refine ⟨suffixResult P p initial last (p.problem.tile pid ordinal),?_,
    suffix_registers P p initial last pid (p.problem.tile pid ordinal) reg,
    suffix_frame P p initial last _ inv⟩
  rw [outer_decomposition,run_append,setup.property,Option.bind_some,run,
    dynamicInner P p setup.val pid registers,ran]
  exact suffixExecution P p initial last _ inv

private theorem sequence_frame (p : Parameters) (one : Nat → State α → Option (State α))
    (ready : State α → Prop) (ids : List Nat) (legal : Nat → Prop)
    (valid : ∀ id∈ids,legal id)
    (single : ∀ id,legal id → ∀ s,ready s →
      ∃ t,one id s=some t ∧ ready t ∧ Frame p s t)
    (s : State α) (hs : ready s) :
    ∃ t,VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence one ids s=
      some t ∧ ready t ∧ Frame p s t := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl,hs,fun _ _ _ => rfl⟩
  | cons id rest ih =>
    obtain ⟨mid,hm,hr,mf⟩ := single id (valid id (by simp)) s hs
    obtain ⟨final,hf,fr,ff⟩ := ih (fun i hi => valid i (by simp [hi])) mid hr
    refine ⟨final,?_,fr,fun r o ho => (ff r o ho).trans (mf r o ho)⟩
    simp only [VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence,
      hm,Option.bind_some,hf]

theorem program_frame (P : Source.Arithmetic α α) (p : Parameters)
    (sms : 0<p.problem.SMS) (pid : Nat) (valid : pid<p.problem.grid) (initial : State α) :
    ∃ final,program P p pid initial=some final ∧ Frame p initial final := by
  open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution in
  let prepared := preludeExecution P p (atProgram initial pid)
  have regs : Registers p pid prepared.val := by
    simpa [VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram]
      using (prelude_invariant P p
        (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram initial pid)).1
  have memory : prepared.val.mem=initial.mem :=
    (prelude_invariant P p
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram initial pid)).2
  have single : ∀ ordinal,ordinal<p.problem.work pid → ∀ current,Registers p pid current →
      ∃ final,Compose.ordinal (backend P) "tile_ordinal" (outer p) ordinal current=some final ∧
        Registers p pid final ∧ Frame p current final := by
    intro ordinal ho current ready
    have tileBound : p.problem.tile pid ordinal<p.problem.tiles :=
      (Source.tiles_exact p.problem sms _).mp
        ((Source.tile_mem p.problem _).mpr ⟨pid,valid,ordinal,ho,rfl⟩)
    exact tile_frame P p current pid ordinal ready tileBound
  obtain ⟨final,ran,_,frame⟩ := sequence_frame p
    (Compose.ordinal (backend P) "tile_ordinal" (outer p)) (Registers p pid)
    (List.range' 0 (p.problem.work pid)) (fun ordinal => ordinal<p.problem.work pid)
    (by intro ordinal ho;simpa [List.mem_range'] using ho) single prepared.val regs
  refine ⟨final,?_,?_⟩
  · unfold program
    change run (backend P) (source p).surfaceBody _=some final
    have rangeAsSequence := Compose.range_sequence (backend P) "tile_ordinal" (outer p)
      0 (p.problem.work pid) prepared.val
    simp only [Int.natCast_zero] at rangeAsSequence
    rw [source_decomposition,run_append,prepared.property,Option.bind_some,run,
      dynamicOuter P p prepared.val pid regs,rangeAsSequence]
    simp only [Nat.sub_zero,ran,run]
    rfl
  · intro r o ho
    exact (frame r o ho).trans (congrFun (congrFun memory r) o)

theorem schedule_frame (P : Source.Arithmetic α α) (p : Parameters)
    (sms : 0<p.problem.SMS) (ids : List Nat) (valid : ∀ pid∈ids,pid<p.problem.grid)
    (s : State α) :
    ∃ final,runSchedule (ids.map (PublicData.job P p)) s=some final ∧ Frame p s final := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,hm,mf⟩ := program_frame P p sms pid (valid pid (by simp)) s
    obtain ⟨final,hf,ff⟩ := ih (fun id hi => valid id (by simp [hi])) mid
    refine ⟨final,?_,fun r o ho => (ff r o ho).trans (mf r o ho)⟩
    change (Invocation.run (PublicData.job P p pid) s).bind
      (runSchedule (rest.map (PublicData.job P p)))=some final
    rw [show (PublicData.job P p pid).run s=some mid from hm]
    exact hf

theorem equivalent (P : Source.Arithmetic α α) (p : Parameters) (sms : 0<p.problem.SMS) :
    (PublicData.io P p).Equivalent (variant P p) := by
  intro vals _ bounds hb initial _
  have safe := PublicData.schedule_safe P p sms vals bounds hb initial
  obtain ⟨final,ran,frame⟩ := schedule_frame P p sms (List.range p.problem.grid)
    (fun pid hp => List.mem_range.mp hp) initial
  have ioFrame : (PublicData.io P p).Frame vals initial final := by
    intro r o ho
    apply frame r o
    by_cases region : r="C"
    · subst r
      right
      intro x
      have h := ho.1 (0 : Fin 1) (x.1,x.2,PUnit.unit) trivial
      simpa [PublicData.write,logicalAddress,Nat.mul_comm] using h
    · exact Or.inl region
  refine ⟨safe,(Pair.schedule_safety_equal P p _ bounds initial).mp safe,
    final,final,ran,(Pair.schedule_equal P p _ initial).symm.trans ran,?_,ioFrame,ioFrame⟩
  intro o j _
  rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PairIO
