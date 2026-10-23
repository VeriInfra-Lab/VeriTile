import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReverseStep
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReversePointers
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelStore

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelMemory KernelReversePointers
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (u : State α) (n : Nat) : Lane → α :=
  KernelReference.reverse M (KernelReference.read c s p u) c.N n
def contraction (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (u : State α) (n : Nat) : α :=
  KernelReference.contraction M (carry M c s p u (n+1))
    ((KernelReference.read c s p u).saved (c.N-2-n))

structure Holds (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤c.N-1
  fixed : Fixed c p u
  sourcePtr : u.regs .ptr [64,16] "S"=some (ptr "S" (off c p (c.N-2) n))
  targetPtr : u.regs .ptr [64,16] "DI"=some (ptr "DI" (off c p (c.N-2) n))
  incomingPtr : u.regs .ptr [64,16] "DS"=some (ptr "DS" (off c p (c.N-1) n))
  gatePtr : u.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gate c p n))
  gradientPtr : u.regs .ptr [] "DG"=some (fun _ : TileIndex [] => (("DG" : RegionName),gradient c p n))
  acc : u.regs .real [64,16] "Dacc"=some (carry M c s p initial n)
  readsSource : ∀ o, u.mem "S" o=initial.mem "S" o
  readsGate : ∀ o, u.mem "D" o=initial.mem "D" o
  readsIncoming : ∀ o, u.mem "DS" o=initial.mem "DS" o
  output : ∀ m, m<n → ∀ (e : Fin 64) (j : Fin 16),
    u.mem "DI" (address c p (c.N-2-m) e.val j.val)=
      KernelReference.stored M .real (carry M c s p initial (m+1) (e,j,PUnit.unit))
  gateOutput : ∀ m, m<n →
    u.mem "DG" (gradientAddress c p (c.N-2-m))=
      KernelReference.stored M .real (contraction M c s p initial m)
  frame : Frame .backward c p initial u

theorem iteration (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (hn : n<c.N-1)
    (inv : Holds M c s p initial n u) :
    ∃ v, run (engine M) (KernelOps.backwardBody c) u=some v ∧
      Holds M c s p initial (n+1) v := by
  let src := off c p (c.N-2) n
  let inc := off c p (c.N-1) n
  let acc := carry M c s p initial n
  let value := KernelReverseStep.value M inc (gate c p n) acc u
  let dg := KernelReverseStep.contraction M src inc (gate c p n) acc u
  let pre := KernelReverseStep.pre M src inc (gate c p n) acc u
  let v := KernelReverseStep.result M c src inc (gate c p n) (gradient c p n) acc u
  have hs : src=(fun e j => address c p (c.N-2-n) e.val j.val) := by
    funext e j
    exact live c p (c.N-2) n (by omega) e j
  have hx : inc=(fun e j => address c p (c.N-1-n) e.val j.val) := by
    funext e j
    exact live c p (c.N-1) n (by omega) e j
  have hg := gate_live c p n (by omega)
  have hgd := gradient_live c p n (by omega)
  have values : value=carry M c s p initial (n+1) := by
    funext i
    simp [value,KernelReverseStep.value,KernelStep.loadedMatrix,KernelStep.loadedScalar,
      KernelOps.next,carry,KernelReference.reverse,KernelReference.update,KernelReference.read,
      raw,inv.readsIncoming,inv.readsGate,acc,hx,hg]
  have gradientValue : dg=contraction M c s p initial n := by
    unfold dg KernelReverseStep.contraction
    change KernelReference.contraction M value (KernelReverseStep.saved src u)=_
    rw [values]
    congr 1
    funext i
    simp [KernelReverseStep.saved,contraction,KernelReference.read,raw,hs,inv.readsSource]
  refine ⟨v,KernelReverseStep.executes M c src inc _ _ acc u inv.sourcePtr inv.targetPtr
    inv.incomingPtr inv.gatePtr inv.gradientPtr inv.acc inv.fixed.numK inv.fixed.numV,
    Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · constructor <;> simp [v,KernelReverseStep.result,KernelReverseStep.written,
      KernelReverseStep.pre,inv.fixed.head,inv.fixed.key,inv.fixed.value,
      inv.fixed.numK,inv.fixed.numV,inv.fixed.offsK,inv.fixed.offsV]
  · simp [v,KernelReverseStep.result,src,off_next]
  · simp [v,KernelReverseStep.result,src,off_next]
  · simp [v,KernelReverseStep.result,inc,off_next]
  · simp [v,KernelReverseStep.result,gate_next]
  · simp [v,KernelReverseStep.result,gradient_next]
  · simp [v,KernelReverseStep.result,KernelReverseStep.written,KernelReverseStep.pre]
    exact values
  · intro o
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i)) (pre.write "DG" (gradient c p n)
      (KernelReference.stored M .real dg))).mem "S" o=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide),State.write_other _ _ _ _ _ _ (Or.inl (by decide))]
    exact inv.readsSource o
  · intro o
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i)) (pre.write "DG" (gradient c p n)
      (KernelReference.stored M .real dg))).mem "D" o=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide),State.write_other _ _ _ _ _ _ (Or.inl (by decide))]
    exact inv.readsGate o
  · intro o
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i)) (pre.write "DG" (gradient c p n)
      (KernelReference.stored M .real dg))).mem "DS" o=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide),State.write_other _ _ _ _ _ _ (Or.inl (by decide))]
    exact inv.readsIncoming o
  · intro m hm e j
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i))
      (pre.write "DG" (gradient c p n) (KernelReference.stored M .real dg))).mem "DI" _=_
    rw [hs]
    by_cases same : m=n
    · subst m
      rw [scatter_cell,values]
      rfl
    · have different : c.N-2-n≠c.N-2-m := by omega
      rw [scatter_other_time c p "DI" .real _ _ different,
        State.write_other _ _ _ _ _ _ (Or.inl (by decide))]
      exact inv.output m (by omega) e j
  · intro m hm
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i)) (pre.write "DG" (gradient c p n)
      (KernelReference.stored M .real dg))).mem "DG" _=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide),hgd]
    by_cases same : m=n
    · subst m
      rw [State.write_same,gradientValue]
    · rw [State.write_other _ _ _ _ _ _ (Or.inr ?_)]
      · exact inv.gateOutput m (by omega)
      · intro equal
        have sameTime := gradient_unique c p _ _ equal
        omega
  · intro r o outside
    simp only [v,KernelReverseStep.result,State.setReg_mem,KernelReverseStep.written]
    change (scatter "DI" .real src (fun i => M.cast (some .fp32) .real .real (value i))
      (pre.write "DG" (gradient c p n) (KernelReference.stored M .real dg))).mem r o=_
    rw [hs,hgd]
    exact (KernelStore.backward_tile_frame c p ⟨c.N-2-n,by omega⟩ _ _ r o outside).trans
      ((KernelStore.backward_scalar_frame c p ⟨c.N-2-n,by omega⟩ _ pre r o outside).trans
        (inv.frame r o outside))

theorem bind_holds (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M c s p initial n u) :
    Holds M c s p initial n (u.setReg "_i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,?_,?_,?_,?_,?_,?_,inv.readsSource,inv.readsGate,inv.readsIncoming,
    inv.output,inv.gateOutput,inv.frame⟩
  · constructor <;> simp [inv.fixed.head,inv.fixed.key,inv.fixed.value,inv.fixed.numK,
      inv.fixed.numV,inv.fixed.offsK,inv.fixed.offsV]
  all_goals simp [inv.sourcePtr,inv.targetPtr,inv.incomingPtr,inv.gatePtr,inv.gradientPtr,inv.acc]

theorem executes (M : Algebra α) (c : Shape) (s : SeedStrides) (p : Program c)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M c s p initial n u) :
    ∃ v, FP.Structural.range (engine M) "_i" n (c.N-1) 1 (KernelOps.backwardBody c) u=some v ∧
      Holds M c s p initial (c.N-1) v := by
  by_cases hn : n<c.N-1
  · obtain ⟨v,he,hv⟩ := iteration M c s p initial n
      (u.setReg "_i" .nat [] (fun _ => n)) hn (bind_holds M c s p initial n u inv)
    obtain ⟨w,hw,hf⟩ := executes M c s p initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [hn,he] using hw
  · have bound := inv.bound
    have eq : n=c.N-1 := by omega
    refine ⟨u,FP.Structural.range_done _ _ _ _ _ _ _ (by omega),by simpa [eq] using inv⟩
termination_by c.N-1-n

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseLoop
