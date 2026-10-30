import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FP
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Real
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open DirectFP NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Actual readable kernel execution for both source versions. -/
def runGrid {α : Type} [Inhabited α] (M : Algebra α) (p : Metadata) (choice : Fin 9)
    (candidate : Bool) : List HostFP.Pid → State α → Option (State α)
  | [],s => some s
  | q::qs,s => do
      let next ← FP.Structural.exec (engine M)
        (if candidate then Kernels.indexCommuted (call p) (tuning choice)
          else Kernels.fp32Source (call p) (tuning choice))
        (HostFP.launch (call p) (tuning choice) q s)
      runGrid M p choice candidate qs next

theorem original_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Metadata) (choice : Fin 9)
    (qs : List HostFP.Pid) (s : State α) :
    runGrid M p choice Bool.false qs s=
      HostFP.run M (call p) (tuning choice) Bool.false qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [runGrid,HostFP.run,Bool.false_eq_true,↓reduceIte,
      Kernels.fp32Source_eq (call p) (tuning choice) rfl rfl]
    congr 1
    funext next
    exact ih next

theorem versions_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Metadata) (choice : Fin 9)
    (qs : List HostFP.Pid) (s : State α) :
    runGrid M p choice Bool.false qs s=runGrid M p choice Bool.true qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [runGrid,Bool.false_eq_true,↓reduceIte]
    rw [KernelRewrite.execution M]
    congr 1
    funext next
    exact ih next

/-- The exact changed candidate statement in the native INT64 setup. -/
def candidateSetup (h : HostCall) (c : Tuning) (q : HostFP.Pid) : NativeScalar.Control :=
  let initial := NativeScalar.setup h c q
  { initial with
    head := NativeWord.enc q.2.2 - NativeWord.enc h.groupCount * initial.chunk }

theorem candidate_setup_eq (h : HostCall) (c : Tuning) (q : HostFP.Pid) :
    candidateSetup h c q=NativeScalar.setup h c q := by
  simp only [candidateSetup,NativeScalar.setup]
  congr 1
  exact congrArg (fun product => NativeWord.enc q.2.2-product) (BitVec.mul_comm _ _)

/-- Native signed-INT64 schedule. The graph's compiler staging uses only node
structure and the selected configuration; input word payloads are never selectors. -/
def native (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s : State Expr) : State Expr :=
  if (p.tensor .a).numel=0 then s else one rest p h a choice s

/-- Empty domains execute an empty program list; nonempty domains use every
actual grid coordinate. This is kernel semantics, without a host return object. -/
def run (rest : Algebra Expr) (p : Metadata) (a : Direct.Allocation p) (choice : Fin 9)
    (candidate : Bool) (s : State Expr) : Option (State Expr) :=
  if (p.tensor .a).numel=0 then some (FPFlat.lift (call p) (Direct.placement p) s)
  else runGrid (FPDefault.backend (algebra rest) (dot choice)) p choice candidate
    (HostFP.grid (call p) (tuning choice)) (FPFlat.lift (call p) (Direct.placement p) s)

/-- Successful actual source and candidate execution, joined to the native
schedule and its typed output stores. This does not equate two failures. -/
theorem success (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s : State Expr) (good : Good s) :
    ∃ final,run rest p a choice Bool.false s=some final ∧
      run rest p a choice Bool.true s=some final ∧
      ((p.tensor .a).numel≠0 →
        (native rest p h a choice s).mem=(FPFlat.publish (Direct.placement p) (call p).Out s final).mem) ∧
      Good (native rest p h a choice s) ∧
      (∀ r off,¬(r=(call p).Out ∧ off<(p.tensor .a).numel) →
        (native rest p h a choice s).mem r off=s.mem r off) := by
  by_cases empty : (p.tensor .a).numel=0
  · exact ⟨FPFlat.lift (call p) (Direct.placement p) s,by simp [run,empty],by simp [run,empty],by simp [empty],
      by simpa [native,empty] using good,by simp [native,empty]⟩
  · have positive : 0<(p.tensor .a).numel := Nat.pos_of_ne_zero empty
    obtain ⟨final,execution,stores⟩ := source_one rest p h a positive choice s good
    refine ⟨final,?_,?_,?_,?_,?_⟩
    · simpa [run,empty,original_eq] using execution
    · simpa [run,empty,←versions_eq,original_eq] using execution
    · intro _; simpa [native,empty] using stores
    · simpa [native,empty] using one_good rest p h a choice s good
    · simpa [native,empty] using one_frame rest p h a positive choice s good

/-- The actual typed source state itself has the independent per-store result
and full exterior frame, before publishing those stores to the native buffer. -/
theorem source_result (rest : Algebra Expr) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (choice : Fin 9) (s final : State Expr)
    (nonempty : (p.tensor .a).numel≠0)
    (success : run rest p a choice Bool.false s=some final) :
    FPGridOutput.Result (algebra rest) (dot choice) (call p) (tuning choice)
      (FPFlat.lift (call p) (Direct.placement p) s) final := by
  obtain ⟨other,execution,result⟩ := FPGridOutput.complete (algebra rest) (dot choice)
    (call p) (tuning choice) (Direct.precision p) (Direct.flat p a)
    (Direct.layout p h a (Nat.pos_of_ne_zero nonempty) (tuning choice))
    (FPFlat.lift (call p) (Direct.placement p) s)
  have same : other=final := by
    have success' : HostFP.run (FPDefault.backend (algebra rest) (dot choice)) (call p)
        (tuning choice) Bool.false (HostFP.grid (call p) (tuning choice))
        (FPFlat.lift (call p) (Direct.placement p) s)=some final := by
      simpa only [run,if_neg nonempty,original_eq] using success
    exact Option.some.inj (execution.symm.trans success')
  simpa only [same] using result

/-- Each logical output has an actual active source writer. Its readback is
an independently defined blocked recurrence over initial input memory. -/
theorem outputs (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s₀ : State Expr) (good : Good s₀)
    (b s g n : Nat) (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    ∃ q∈HostFP.grid (call p) (tuning choice),
      ∃ i : TileIndex [(tuning choice).BM,(tuning choice).BN],
        FPReference.active (HostFP.launch (call p) (tuning choice) q s₀) (tuning choice).BM (tuning choice).BN
          (HostContract.parameters (call p)) i ∧
        FPFlat.outOffset (call p) (tuning choice) q s₀ i=Direct.tensorIndex p b s g n ∧
        (native rest p h a choice s₀).mem (call p).Out (Direct.tensorIndex p b s g n)=
          .mk .fp32 (FPReference.expected (algebra rest) (dot choice) (tuning choice).BM (tuning choice).BN
            (tuning choice).BCS (HostContract.parameters (call p))
            (HostFP.launch (call p) (tuning choice) q (FPFlat.lift (call p) (Direct.placement p) s₀)) i) := by
  have bound := Direct.tensor_bound p h b s g n hb hs hg hn
  have positive : 0<(p.tensor .a).numel := Nat.zero_lt_of_lt bound
  obtain ⟨q,hq,i,hi,address,_⟩ := Direct.coordinate_covered p h ⟨fun _ => 0,fun _ => 0,fun _ => 0,fun _ => 0⟩ choice b s g n hb hs hg hn
  rw [←Direct.grid_eq p h positive] at hq
  rw [←Direct.parameters_eq p h positive] at hi address
  let q' : HostFP.Pid := (q.tile,q.batch,q.chunkHead)
  have scheduled : q'∈HostFP.grid (call p) (tuning choice) := by
    have bounds := (HostGrid.mem_programs _ _).mp hq
    simpa [q',HostFP.grid,HostContract.grid,HostGrid.dimensions,HostContract.parameters] using bounds
  have active : FPReference.active (HostFP.launch (call p) (tuning choice) q' s₀) (tuning choice).BM (tuning choice).BN
      (HostContract.parameters (call p)) i := hi
  have offset : FPFlat.outOffset (call p) (tuning choice) q' s₀ i=Direct.tensorIndex p b s g n := address
  refine ⟨q',scheduled,i,active,offset,?_⟩
  simpa only [native,if_neg (Nat.ne_of_gt positive),offset] using
    one_readback rest p h a positive choice s₀ good q' scheduled i active

#axiomsClean candidate_setup_eq
#axiomsClean original_eq
#axiomsClean versions_eq
#axiomsClean success
#axiomsClean source_result
#axiomsClean outputs
#specNonCircular FPReference.expected avoiding [run,native,Kernels.fp32Source]
end
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelFP
