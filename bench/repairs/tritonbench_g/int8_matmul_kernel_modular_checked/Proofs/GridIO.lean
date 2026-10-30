import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private theorem launch_initial (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (pid : Nat) (initial : State Word32) :
    (launch cast rest layout s c pid).initial initial =
      programState initial (flatten initial) pid := by
  simp [launch,Invocation.initial,bindRegisters,programState,tileState,flatten]

private theorem same_binding (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (xs : Inputs layout s)
    (initial current : State Word32)
    (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (same : SameInputs layout initial (flatten current)) :
    (matmulIO cast rest layout s c).InputBinding xs current := by
  intro port i active
  rcases port with ⟨port,hp⟩
  have hp' : port<2 := hp
  interval_cases port
  · exact (congrArg (Cell.read .real) (same _ _ (Or.inl rfl))).trans (hin ⟨0,hp⟩ i active)
  · exact (congrArg (Cell.read .nat) (same _ _ (Or.inr rfl))).trans (hin ⟨1,hp⟩ i active)

private theorem frozen_same (layout : Layout) (s : Shape) (c : Fin 15)
    (initial current : State Word32) (same : SameInputs layout initial (flatten current))
    (pid : Nat) (memory : Backing) :
    frozenStep layout s c current pid memory=frozenStep layout s c initial pid memory := by
  have ha : inputWords layout current=inputWords layout initial := by
    funext m k
    exact congrArg (Cell.read .real) (same _ _ (Or.inl rfl))
  have hb : packedWords layout current=packedWords layout initial := by
    funext n q
    exact congrArg (fun cell => BitVec.ofNat 32 (cell.read .nat)) (same _ _ (Or.inr rfl))
  have writes : tileWrites layout s c current pid=tileWrites layout s c initial pid := by
    rfl
  funext loc
  change (if tileWrites layout s c current pid loc then denseCell layout s current loc else memory loc) =
    (if tileWrites layout s c initial pid loc then denseCell layout s initial loc else memory loc)
  change (if tileWrites layout s c initial pid loc then denseCell layout s current loc else memory loc) = _
  simp only [denseCell,matrixWordTarget,ha,hb]

/-- Full actual invocation schedule, using the prior per-program execution
certificate and its frozen-write characterization on each predecessor state. -/
theorem schedule_correct (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs : Inputs layout s) (domain : Domain layout s xs) (bounds : RegionName → Nat)
    (hb : (matmulIO cast rest layout s c).InBounds bounds xs)
    (initial : State Word32) (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (mpos : 0<s.m) (npos : 0<s.n) (hout : OutputLayout layout s)
    (separate : layout.output≠layout.input ∧ layout.output≠layout.weight)
    (schedule : List Nat) (current : State Word32)
    (same : SameInputs layout initial (flatten current)) :
    ScheduleSafe bounds (schedule.map (launch cast rest layout s c)) current ∧
    ∃ final,runSchedule (schedule.map (launch cast rest layout s c)) current=some final ∧
      flatten final=Schedule.run (frozenStep layout s c initial) schedule (flatten current) := by
  induction schedule generalizing current with
  | nil => exact ⟨trivial,current,rfl,rfl⟩
  | cons pid tail ih =>
    have binding := same_binding cast rest layout s c xs initial current hin same
    have dom := domains cast rest layout s c xs current binding domain
    obtain ⟨next,ran,mem⟩ := program_frozen cast castLeaf rest layout s c legal current
      mpos npos dom.1 dom.2 hout (flatten current) (fun _ _ _ => rfl) pid
    have actual : (launch cast rest layout s c pid).run current=some next := by
      change FP.Structural.exec (backend cast rest) (source (parameters layout s c))
        ((launch cast rest layout s c pid).initial current)=some next
      rw [launch_initial]
      exact ran
    rw [frozen_same layout s c initial current same pid] at mem
    have follows := ih next (by
      rw [mem]
      exact frozen_inputs layout s c initial separate (flatten current) same pid)
    constructor
    · change (launch cast rest layout s c pid).Safe bounds current ∧ _
      constructor
      · apply source_bounds cast rest layout s c legal xs bounds hb
          ((launch cast rest layout s c pid).initial current) _ mpos npos hout
        exact dom.1
      · intro t ht
        rw [actual] at ht
        cases ht
        exact follows.1
    · obtain ⟨final,hr,hmem⟩ := follows.2
      refine ⟨final,?_,?_⟩
      · simpa only [List.map_cons,runSchedule,actual,Option.bind_some] using hr
      · simpa only [Schedule.run,List.foldl_cons,mem] using hmem

private theorem empty_grid (s : Shape) (c : Fin 15) (empty : s.m=0 ∨ s.n=0) :
    s.programs c=0 := by
  have bm : 0<(configs c).bm := by have := config_bounds c; omega
  have bn : 0<(configs c).bn := by have := config_bounds c; omega
  rcases empty with hm|hn
  · simp [Shape.programs,Shape.rows,hm,ceil_zero (configs c).bm bm]
  · simp [Shape.programs,Shape.cols,hn,ceil_zero (configs c).bn bn]

/-- Shared IO guarantees actual complete-grid safety and execution, all signed
INT32 outputs, and the complete exterior frame. -/
theorem implements (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    (matmulIO cast rest layout s c).ImplementsOn (Domain layout s) (formula layout s) := by
  intro xs domain bounds hb initial hin
  have complete : ScheduleSafe bounds ((matmulIO cast rest layout s c).schedule xs) initial ∧
      ∃ final,runSchedule ((matmulIO cast rest layout s c).schedule xs) initial=some final ∧
        flatten final=gridMemory layout s c initial := by
    by_cases empty : s.m=0 ∨ s.n=0
    · have hz := empty_grid s c empty
      exact ⟨by simp [matmulIO,hz,ScheduleSafe],initial,by simp [matmulIO,hz,runSchedule],
        by simp [gridMemory,hz,Schedule.run]⟩
    · exact schedule_correct cast castLeaf rest layout s c legal xs domain bounds hb initial hin
        (by omega) (by omega) hout separate _ initial (fun _ _ _ => rfl)
  obtain ⟨safe,final,ran,hmem⟩ := complete
  refine ⟨safe,final,ran,?_,?_⟩
  · intro port i _
    have port0 : port=⟨0,by simp [matmulIO]⟩ := by
      apply Fin.ext
      change port.val=0
      have hp : port.val<1 := port.isLt
      omega
    subst port
    change mapCell id (final.mem layout.output (i.1.val*max s.n 1+i.2.1.val)) = _
    have hv := congrFun hmem (layout.output,i.1.val*max s.n 1+i.2.1.val)
    have value := grid_values layout s c initial hout i.1.val i.2.1.val i.1.isLt i.2.1.isLt
    change gridMemory layout s c initial (layout.output,i.1.val*max s.n 1+i.2.1.val)=_ at value
    rw [value] at hv
    change final.mem layout.output (i.1.val*max s.n 1+i.2.1.val)=_ at hv
    rw [hv]
    change Cell.mk .int (matrixWordTarget layout s initial i.1.val i.2.1.val).toInt = _
    rw [target_binding cast rest layout s c xs initial hin i.1 i.2.1]
    rfl
  · intro r off outside
    have ext : r≠layout.output ∨ s.m*s.n≤off := by
      by_cases hr : r=layout.output
      · right
        by_contra hn
        have hlt : off<s.m*s.n := by omega
        have npos : 0<s.n := by nlinarith
        have mr : off/s.n<s.m := (Nat.div_lt_iff_lt_mul npos).mpr hlt
        have nr : off%s.n<s.n := Nat.mod_lt _ npos
        have miss := outside.1 ⟨0,by simp [matmulIO]⟩
          (⟨off/s.n,mr⟩,⟨off%s.n,nr⟩,PUnit.unit) trivial
        rcases miss with hreg|hoff
        · exact hreg hr
        · apply hoff
          change off=off/s.n*max s.n 1+off%s.n
          rw [max_eq_left (show 1≤s.n by omega)]
          simpa [Nat.mul_comm,Nat.add_comm] using (Nat.mod_add_div off s.n).symm
      · exact Or.inl hr
    exact (congrFun hmem (r,off)).trans (grid_frame layout s c initial hout (r,off) ext)

#axiomsClean schedule_correct
#axiomsClean implements
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
