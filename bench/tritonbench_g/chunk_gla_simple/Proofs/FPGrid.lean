import bench.tritonbench_g.chunk_gla_simple.Proofs.FPComplete
import bench.tritonbench_g.chunk_gla_simple.Proofs.Layout

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPGrid
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open FPValue
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

abbrev Pid (c : Shape) := GridIndex (grid c)
abbrev launch := @HostFP.atProgram

def run (M : Algebra α) (c : Shape) (scale : ℝ) : List (Pid c) → State α → Option (State α)
  | [],s => some s
  | q::rest,s => do
    let t ← FP.Structural.exec (engine M) (kernel c .fp32 .fp32 .fp32 .fp32 scale) (launch c s q)
    run M c scale rest t

/-- Independent recurrence of masked stores, with no interpreter in its definition. -/
def reference (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c) :
    List (Pid c) → State α → State α
  | [],s => s
  | q::rest,s => reference M c scale d rest (written M c scale d (launch c s q))

theorem output_congr (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (s t : State α) (hp : s.pids=t.pids)
    (hi : ∀ r ∈ (["q","k","h","v","g"] : List RegionName), s.mem r=t.mem r) :
    outputValue M c d s scale=outputValue M c d t scale := by
  have hq := hi "q" (by simp)
  have hk := hi "k" (by simp)
  have hh := hi "h" (by simp)
  have hv := hi "v" (by simp)
  have hg := hi "g" (by simp)
  have qv (n : Nat) : FPValue.qValue M c s n=FPValue.qValue M c t n := by
    funext i
    simp only [FPValue.qValue,qOffset,FPValue.row,hp,hq]
  have kv (n : Nat) : FPValue.kValue M c s n=FPValue.kValue M c t n := by
    funext i
    simp only [FPValue.kValue,kOffset,FPValue.row,hp,hk]
  have hhV (n : Nat) : FPValue.hValue M c s n=FPValue.hValue M c t n := by
    funext i
    simp only [FPValue.hValue,hOffset,FPValue.column,hp,hh]
  have vv : FPValue.vValue M c s=FPValue.vValue M c t := by
    funext i
    simp only [FPValue.vValue,FPValue.active,oOffset,FPValue.row,FPValue.column,hp,hv]
  have gv : FPValue.gValue M c s=FPValue.gValue M c t := by
    funext i
    simp only [FPValue.gValue,gOffset,FPValue.row,hp,hg]
  have hu : update M c d s=update M c d t := by
    funext n a
    simp only [FPValue.update,qv,kv,hhV]
  funext i
  simp only [FPValue.outputValue,FPValue.accumulated,hu,FPValue.gatedO,FPValue.gatedS,
    gv,vv]

theorem written_mem_congr (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (s t : State α) (hm : s.mem=t.mem) (hp : s.pids=t.pids) :
    (written M c scale d s).mem=(written M c scale d t).mem := by
  have hv := output_congr M c scale d s t hp (fun r _ => congrFun hm r)
  simp only [written,VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    FPValue.active,oOffset,FPValue.row,FPValue.column,hp,hm,hv]

theorem reference_mem_congr (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (jobs : List (Pid c)) (s t : State α) (hm : s.mem=t.mem) :
    (reference M c scale d jobs s).mem=(reference M c scale d jobs t).mem := by
  induction jobs generalizing s t with
  | nil => exact hm
  | cons q rest ih => exact ih _ _ (written_mem_congr M c scale d _ _ hm rfl)

theorem run_memory (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (jobs : List (Pid c)) (s : State α) :
    ∃ t, run M c scale jobs s=some t ∧ t.mem=(reference M c scale d jobs s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons q rest ih =>
    obtain ⟨mid,he,hm⟩ := source_memory M c scale d (launch c s q)
    obtain ⟨t,ht,hmt⟩ := ih mid
    refine ⟨t,?_,hmt.trans (reference_mem_congr M c scale d rest _ _ hm)⟩
    simp only [run,he,bind,Option.bind_some,ht]

theorem reference_frame (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (jobs : List (Pid c)) (s : State α) (r : RegionName) (off : Nat)
    (miss : r≠"o" ∨ ∀ q ∈ jobs, ∀ i, active c (launch c s q) i → off≠oOffset c (launch c s q) i) :
    (reference M c scale d jobs s).mem r off=s.mem r off := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    rw [reference,ih]
    · exact frame M c scale d (launch c s q) r off
        (miss.imp_right (fun h i hi => h q (by simp) i hi))
    · exact miss.imp_right (fun h u hu i hi => h u (List.mem_cons_of_mem _ hu) i hi)

theorem geometry (c : Shape) (s t : State α) (q q' : Pid c) (i j : OIndex c)
    (hi : active c (launch c s q) i) (hj : active c (launch c t q') j)
    (he : oOffset c (launch c s q) i=oOffset c (launch c t q') j) : q=q' ∧ i=j := by
  have hh := Host.address_injective c (launchIndex c q) (launchIndex c q') i j
  have hp : ∀ (s : State α) (q : Pid c),
      (launch c s q).pids 0=(launchIndex c q).value ∧
      (launch c s q).pids 1=(launchIndex c q).chunk ∧
      (launch c s q).pids 2=(launchIndex c q).head := by
    intro s q
    simp [launch,HostFP.atProgram,GridIndex.toPids,launchIndex,grid,Shape.grid,Grid.rank]
  obtain ⟨hl,hij⟩ := hh
    (by simpa only [FPValue.active,FPValue.row,FPValue.column,(hp s q).1,(hp s q).2.1,
      Math.active,layout,Math.row,Math.column] using hi)
    (by simpa only [FPValue.active,FPValue.row,FPValue.column,(hp t q').1,(hp t q').2.1,
      Math.active,layout,Math.row,Math.column] using hj)
    (by simpa only [oOffset,FPValue.row,FPValue.column,(hp s q).1,(hp s q).2.1,(hp s q).2.2,
      (hp t q').1,(hp t q').2.1,(hp t q').2.2,Math.outputAddress,layout,Math.row,Math.column] using he)
  exact ⟨launchIndex_injective c hl,hij⟩

theorem reference_readback (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (jobs : List (Pid c)) (s : State α) (hn : jobs.Nodup) (q : Pid c) (hq : q∈jobs)
    (i : OIndex c) (hi : active c (launch c s q) i) :
    (reference M c scale d jobs s).mem "o" (oOffset c (launch c s q) i)=
      .mk .fp32 (outputValue M c d (launch c s q) scale i) := by
  induction jobs generalizing s with
  | nil => simp at hq
  | cons u rest ih =>
    let mid := written M c scale d (launch c s u)
    obtain ⟨hu,hn⟩ := List.nodup_cons.mp hn
    by_cases hqu : q=u
    · subst q
      rw [reference,reference_frame]
      · exact readback M c scale d (launch c s u) i hi
          (fun a b ha hb he => (geometry c s s u u a b ha hb he).2)
      · right
        intro v hv j hj he
        have heq := (geometry c s mid u v i j hi hj he).1
        exact hu (heq ▸ hv)
    · have hqr : q∈rest := by simpa [hqu] using hq
      have hr := ih mid hn hqr hi
      change (reference M c scale d rest mid).mem "o" (oOffset c (launch c mid q) i)=_
      rw [hr]
      apply congrArg (Cell.mk .fp32)
      apply congrFun
      apply output_congr M c scale d (launch c mid q) (launch c s q) rfl
      intro r hr
      funext off
      exact frame M c scale d (launch c s u) r off (Or.inl (by
        simp only [List.mem_cons,List.mem_nil_iff,or_false] at hr
        rcases hr with rfl|rfl|rfl|rfl|rfl <;> decide))

def Result (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (s t : State α) : Prop :=
  (∀ q : Pid c, ∀ i, active c (launch c s q) i →
    t.mem "o" (oOffset c (launch c s q) i)=.mk .fp32 (outputValue M c d (launch c s q) scale i)) ∧
  (∀ (r : RegionName) off, (r≠"o" ∨ ∀ q : Pid c, ∀ i, active c (launch c s q) i →
    off≠oOffset c (launch c s q) i) → t.mem r off=s.mem r off)

def programs (c : Shape) : List (Pid c) := (Finset.univ : Finset (Pid c)).toList

theorem programs_nodup (c : Shape) : (programs c).Nodup := Finset.nodup_toList _
theorem mem_programs (c : Shape) (q : Pid c) : q∈programs c := by simp [programs]

/-- Every actual program executes into one merged state; independent initial
input readback, exact FP32 tags, and the full tagged-memory frame all compose. -/
theorem complete (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (jobs : List (Pid c)) (hn : jobs.Nodup) (hc : ∀ q, q∈jobs) (s : State α) :
    ∃ t, run M c scale jobs s=some t ∧ Result M c scale d s t := by
  obtain ⟨t,he,hm⟩ := run_memory M c scale d jobs s
  refine ⟨t,he,?_,?_⟩
  · intro q i hi
    rw [hm]
    exact reference_readback M c scale d jobs s hn q (hc q) i hi
  · intro r off hh
    rw [hm]
    exact reference_frame M c scale d jobs s r off (hh.imp_right (fun h q _ => h q))

theorem whole_grid (M : Algebra α) (c : Shape) (scale : ℝ) (d : Typed.Dots M c) (s : State α) :
    ∃ t, run M c scale (programs c) s=some t ∧ Result M c scale d s t :=
  complete M c scale d (programs c) (programs_nodup c) (mem_programs c) s

theorem returned (M : Algebra α) (c : Shape) (ht : 0<c.BT) (scale : ℝ)
    (d : Typed.Dots M c) (s t : State α) (hr : Result M c scale d s t)
    (h : Fin c.heads) (r : Fin c.T) (v : Fin c.V) :
    t.mem "o" (h.val*(c.T*c.V)+r.val*c.V+v.val)=
      .mk .fp32 (outputValue M c d (launch c s (owner c ht h r v)) scale (lane c ht r v)) := by
  have ha : active c (launch c s (owner c ht h r v)) (lane c ht r v) := by
    simpa [FPValue.active,FPValue.row,FPValue.column,launch,HostFP.atProgram,owner,index,
      GridIndex.toPids,grid,Shape.grid,Grid.rank,lane,Nat.div_add_mod'] using And.intro r.isLt v.isLt
  have hh := hr.1 (owner c ht h r v) (lane c ht r v) ha
  simpa [oOffset,FPValue.row,FPValue.column,launch,HostFP.atProgram,owner,index,
    GridIndex.toPids,grid,Shape.grid,Grid.rank,lane,Nat.div_add_mod'] using hh

#specNonCircular Result avoiding [run,FP.Structural.exec]
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPGrid
