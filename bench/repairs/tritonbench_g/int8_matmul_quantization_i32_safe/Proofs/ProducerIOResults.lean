import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIOSafety

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

/-- Exact original launch setup, including untouched local state and PID axes. -/
theorem initial_eq (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (s : State α) :
    (invocation M integer8 p r).initial s=KernelProducerGrid.atRow s r := by
  simp [invocation,Invocation.initial,bindRegisters,KernelProducerGrid.atRow]
  funext ax
  by_cases h : ax=0 <;> simp [h,Function.update]

theorem invocation_run (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (s : State α) :
    (invocation M integer8 p r).run s=FP.Structural.exec (KernelProducer.backend M integer8)
      (KernelSource.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p)))
      (KernelProducerGrid.atRow s r) := by
  unfold Invocation.run
  rw [initial_eq]
  rfl

theorem schedule_eq (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (order : List (Fin p.rows)) (s : State α) :
    runSchedule (order.map (invocation M integer8 p)) s=KernelProducerGrid.run M integer8 p order s := by
  induction order generalizing s with
  | nil=>rfl
  | cons r rest ih=>
    simp only [List.map,runSchedule,invocation_run,KernelProducerGrid.run,ih]
    rfl

/-- The full original arbitrary row order is checked through actual successor
states, including an empty row grid. -/
theorem schedule_safe (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (order : I32Safe.Producer.Schedule p) (xs : Inputs α p) (bounds : RegionName→Nat)
    (bounded : (io M integer8 p order).InBounds bounds xs) (s : State α) :
    ScheduleSafe bounds ((io M integer8 p order).schedule xs) s := by
  dsimp only [ProgramIO.InBounds,io,layout] at bounded
  have each : ∀ r : Fin p.rows,∀ s : State α,(invocation M integer8 p r).Safe bounds s := by
    intro r s
    change TraceSafe (KernelProducer.backend M integer8) bounds
      (Source.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p)))
      ((invocation M integer8 p r).initial s)
    rw [initial_eq]
    apply source_safe M integer8 p r s bounds
    · intro c hc
      exact bounded.2.1 0 (r,⟨c.val,hc⟩,()) trivial
    · intro c hc
      exact bounded.2.2.1 0 (r,⟨c.val,hc⟩,()) trivial
    · exact bounded.2.2.1 1 (r,()) trivial
  change ScheduleSafe bounds (order.order.map (invocation M integer8 p)) s
  generalize order.order=rows
  induction rows generalizing s with
  | nil=>trivial
  | cons r rest ih=>exact ⟨each r s,fun _ _=>ih _⟩

theorem reads (M : Algebra α) (integer8 : α→BitVec 8) (zero : α) (p : Geometry)
    (order : I32Safe.Producer.Schedule p) (xs : Inputs α p) (s : State α)
    (binding : (io M integer8 p order).InputBinding xs s) :
    KernelProducerTargets.Reads p (values zero p xs) s := by
  dsimp only [ProgramIO.InputBinding,io,layout] at binding
  intro r c hr hc
  have h:=binding 0 (⟨r,hr⟩,⟨c,hc⟩,()) trivial
  simpa only [values,dif_pos hr,dif_pos hc] using h

/-- Every raw Real producer state supplies admitted finite logical input values. -/
def inputsOfState (p : Geometry) (s : State ℝ) : Inputs ℝ p :=
  fun port=>match port with
  | ⟨0,_⟩=>fun j=>(s.mem "x" (j.1.val*p.columns+j.2.1.val)).read .real

theorem inputsOfState_binding (M : Algebra ℝ) (integer8 : ℝ→BitVec 8) (p : Geometry)
    (order : I32Safe.Producer.Schedule p) (s : State ℝ) :
    (io M integer8 p order).InputBinding (inputsOfState p s) s := by
  intro port j _
  fin_cases port
  rfl

theorem outside_exterior (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (order : I32Safe.Producer.Schedule p) (xs : Inputs α p) (r : RegionName) (off : Nat)
    (outside : (io M integer8 p order).Outside xs r off) :
    (r≠"q" ∨ p.rows*p.columns≤off) ∧ (r≠"scales" ∨ p.rows≤off) := by
  dsimp only [ProgramIO.Outside,io,layout] at outside
  constructor
  · by_cases hr : r="q"
    · right
      by_contra large
      have hbound : off<p.rows*p.columns:=Nat.lt_of_not_ge large
      have hc : 0<p.columns:=by nlinarith
      have row : off/p.columns<p.rows:=(Nat.div_lt_iff_lt_mul hc).mpr hbound
      have col : off%p.columns<p.columns:=Nat.mod_lt _ hc
      have bad:=outside.1 0 (⟨off/p.columns,row⟩,⟨off%p.columns,col⟩,()) trivial
      have address : off=off/p.columns*p.columns+off%p.columns:=by
        have h:=Nat.mod_add_div off p.columns
        nlinarith
      exact bad.elim (fun h=>h hr) (fun h=>h address)
    · exact Or.inl hr
  · by_cases hr : r="scales"
    · right
      by_contra large
      have hbound : off<p.rows:=Nat.lt_of_not_ge large
      exact (outside.1 1 (⟨off,hbound⟩,()) trivial).elim (fun h=>h hr) (fun h=>h rfl)
    · exact Or.inl hr

theorem real_result (p : Geometry) (selection : I32Safe.Producer.Selection p)
    (covers : p.columns≤p.width) (rest : Algebra ℝ) (order : I32Safe.Producer.Schedule p)
    (xs : Inputs ℝ p) (s : State ℝ) (binding : (realIO p selection rest order).InputBinding xs s) :
    ∃ final,runSchedule ((realIO p selection rest order).schedule xs) s=some final ∧
      (realIO p selection rest order).OutputBinding xs (mathematical p xs) final ∧
      (realIO p selection rest order).Frame xs s final := by
  obtain ⟨final,ran,data,scales,frame⟩:=I32Safe.Producer.mathematical_complete p selection covers rest
    (values 0 p xs) s (reads _ _ 0 p order xs s binding) order
  refine ⟨final,(schedule_eq _ _ p order.order s).trans ran,?_,?_⟩
  · intro port j _
    rcases port with ⟨port,hp⟩
    have small : port<2:=hp
    interval_cases port
    · exact congrArg (mapCell id) (data j.1 ⟨j.2.1.val,Nat.lt_of_lt_of_le j.2.1.isLt covers⟩ j.2.1.isLt)
    · exact congrArg (mapCell id) (scales j.1)
  · intro r off outside
    have h:=outside_exterior _ _ p order xs r off outside
    exact frame r off h.1 h.2

theorem real_implements (p : Geometry) (selection : I32Safe.Producer.Selection p)
    (covers : p.columns≤p.width) (rest : Algebra ℝ) (order : I32Safe.Producer.Schedule p) :
    (realIO p selection rest order).Implements (mathematical p) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded s _
    exact schedule_safe _ _ p order xs bounds bounded s
  · exact real_result p selection covers rest order

local instance kernelInhabitedWord32 : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

theorem native_result (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : I32Safe.Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra Word32) (order : I32Safe.Producer.Schedule p)
    (xs : Inputs Word32 p) (s : State Word32) (binding : (nativeIO f p selection rest order).InputBinding xs s) :
    ∃ final,runSchedule ((nativeIO f p selection rest order).schedule xs) s=some final ∧
      (nativeIO f p selection rest order).OutputBinding xs (nativeFormula f p selection xs) final ∧
      (nativeIO f p selection rest order).Frame xs s final := by
  obtain ⟨final,ran,data,scales,frame⟩:=I32Safe.Producer.native_complete f p selection covers rest
    (values (FP.Value.zero .fp32) p xs) s (reads _ _ (FP.Value.zero .fp32) p order xs s binding) order
  refine ⟨final,(schedule_eq _ _ p order.order s).trans ran,?_,?_⟩
  · intro port j _
    rcases port with ⟨port,hp⟩
    have small : port<2:=hp
    interval_cases port
    · exact congrArg (mapCell id) (data j.1 ⟨j.2.1.val,Nat.lt_of_lt_of_le j.2.1.isLt covers⟩ j.2.1.isLt)
    · exact congrArg (mapCell id) (scales j.1)
  · intro r off outside
    have h:=outside_exterior _ _ p order xs r off outside
    exact frame r off h.1 h.2

theorem native_implements (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : I32Safe.Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra Word32) (order : I32Safe.Producer.Schedule p) :
    (nativeIO f p selection rest order).Implements (nativeFormula f p selection) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded s _
    exact schedule_safe _ _ p order xs bounds bounded s
  · exact native_result f p selection covers rest order

#axiomsClean schedule_eq
#axiomsClean schedule_safe
#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
