import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.Geometry
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPRow
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Local numerical operations, not equations or complete-output callbacks.
The scan retains its kind, axis, direction, precision and ordered input lanes. -/
structure Ops (α : Type) where
  scalar : Algebra α
  scan : Option ComputeDType → {shape : TileShape} → ScanOp →
    (ax : Fin shape.length) → ScanDirection → Values α .real shape → Values α .real shape
  sum : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (ax : Fin shape.length) → (keep : Bool) → Values α dtype.toTileDType shape →
    Values α dtype.toTileDType (TileShape.reduceShape shape ax keep)

def Ops.algebra {α : Type} (o : Ops α) : Algebra α :=
  {o.scalar with
    scan := fun p {_} k a d v => some (o.scan p k a d v)
    reduceSumTyped := fun p d {_} a k v => some (o.sum p d a k v)}

def parts (dtype outType : FloatDType) (input output : RegionName) (T BT : Nat) : Source.IntRange :=
  (Source.intRange? (repairedKernel dtype outType input output T BT).surfaceBody "i_t").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem split_source (dtype outType : FloatDType) (input output : RegionName) (T BT : Nat) :
    (repairedKernel dtype outType input output T BT).surfaceBody =
      (parts dtype outType input output T BT).before ++
        [.forRangeInt "i_t" (parts dtype outType input output T BT).start
          (parts dtype outType input output T BT).stop
          (parts dtype outType input output T BT).stride
          (parts dtype outType input output T BT).body] := rfl

def Ready {α : Type} (pid : Nat) (s : State α) : Prop :=
  s.regs .nat [] "i_bh"=some (fun _ => pid) ∧ (s.regs .real [] "b_z").isSome

def Frame {α : Type} (output : RegionName) (pid T : Nat) (s t : State α) : Prop :=
  ∀ rg off, (rg≠output ∨ ∀ i, i<T → off≠pid*T+i) → t.mem rg off=s.mem rg off

/-- All prelude registers are constructed from the actual row PID. -/
theorem prelude {α : Type} [Inhabited α] (o : Ops α)
    (input output : RegionName) (T BT : Nat) (s : State α) :
    ∃ t, run (engine o.algebra) (parts .fp32 .fp32 input output T BT).before s=some t ∧
      Ready (s.pids 0) t ∧ t.mem=s.mem := by
  simp (config := {maxSteps := 100000}) [parts,repairedKernel,
    ComputeKernel.surfaceBody,Source.intRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,Ops.algebra,ComputeDType.eraseDType,numeric,numericLt,bop,State.setReg,Ready]

@[simp] theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).regs=s.regs := by
  rw [State.masked_scatter_eq_filter,State.scatter_regs]

@[simp] theorem int_address (p T j BT lane : Nat) :
    ((p:Int)*T+((j:Int)*BT+lane)).toNat=p*T+(j*BT+lane) := by omega

@[simp] theorem int_mask (j BT lane T : Nat) :
    decide ((j:Int)*BT+lane<(T:Int))=decide (j*BT+lane<T) := by norm_cast

@[simp] theorem int_lt (j BT lane T : Nat) :
    ((j:Int)*BT+lane<(T:Int)) ↔ j*BT+lane<T := by norm_cast

/-- Every chunk executes the repair's signed pointer loads, scan, sum, carry and store.
The arbitrary unspecified tail is preserved; only masked output cells change. -/
theorem body {α : Type} [Inhabited α] (o : Ops α)
    (input output : RegionName) (T BT pid : Nat)
    (s : State α) (ready : Ready pid s) (index : Nat) :
    ∃ t, run (engine o.algebra) (parts .fp32 .fp32 input output T BT).body
        (s.setReg "i_t" .int [] (fun _ => (index : Int)))=some t ∧
      Ready pid t ∧ Frame output pid T s t := by
  obtain ⟨hp,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  all_goals
    simp (config := {maxSteps := 300000}) [parts,repairedKernel,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,Ops.algebra,ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,Ready,Frame,hp,hz,BlockPtr.inBounds,BlockPtr.address,TileShape.indexToList]
  all_goals
    intro rg off h
    apply State.masked_scatter_frame
    rcases h with hn | hn
    · exact Or.inl hn
    · right
      intro i _ hi
      exact hn (index*BT+i.1.val) hi

/-- Constructive finite source loop, retaining the exact carry register and
arbitrary lane values returned by the local numerical operations. -/
theorem loop {α : Type} [Inhabited α] (o : Ops α)
    (input output : RegionName) (T BT pid cur stop : Nat)
    (s : State α) (ready : Ready pid s) :
    ∃ t, rangeInt (engine o.algebra) "i_t" cur stop 1 (parts .fp32 .fp32 input output T BT).body s=some t ∧
      Ready pid t ∧ Frame output pid T s t := by
  by_cases hlt : cur<stop
  · obtain ⟨mid,hm,hr,hf⟩ := body o input output T BT pid s ready cur
    obtain ⟨t,ht,htr,htf⟩ := loop o input output T BT pid (cur+1) stop mid hr
    refine ⟨t,?_,htr,?_⟩
    · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hlt),hm]
      exact ht
    · intro rg off hn
      exact (htf rg off hn).trans (hf rg off hn)
  · refine ⟨s,?_,ready,fun _ _ _ => rfl⟩
    rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast hlt)]
termination_by stop-cur
 decreasing_by omega

/-- The complete repair body succeeds for every source memory and
storage branch. No whole-kernel success or numerical equality is a premise. -/
theorem complete {α : Type} [Inhabited α] (o : Ops α)
    (input output : RegionName) (T BT : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (engine o.algebra) (repairedKernel .fp32 .fp32 input output T BT) s=some t ∧
      Frame output (s.pids 0) T s t := by
  obtain ⟨initial,hpre,hready,hmem⟩ := prelude o input output T BT s
  obtain ⟨final,hl,_,hf⟩ := loop o input output T BT (s.pids 0)
    0 ((T+BT-1)/BT) initial hready
  refine ⟨final,?_,?_⟩
  · change run (engine o.algebra) (repairedKernel .fp32 .fp32 input output T BT).surfaceBody s=some final
    rw [split_source .fp32 .fp32,run_append,hpre]
    simp only [Option.bind_some,run]
    have step_eq : step (engine o.algebra)
        (.forRangeInt "i_t" (parts .fp32 .fp32 input output T BT).start
          (parts .fp32 .fp32 input output T BT).stop (parts .fp32 .fp32 input output T BT).stride
          (parts .fp32 .fp32 input output T BT).body) initial=some final := by
      simp only [step]
      change rangeInt (engine o.algebra) "i_t" 0 (((T+BT-1)/BT : Nat) : Int) 1
        (parts .fp32 .fp32 input output T BT).body initial=some final
      exact hl
    rw [step_eq]
    rfl
  · intro rg off hn
    rw [hf rg off hn,hmem]

#axiomsClean prelude
#axiomsClean body
#axiomsClean loop
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPRow
