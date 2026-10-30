import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealStep
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealInit

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealExecution
open VeriTile.Triton RealValues RealProgram
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def accumulated (h : Call) (c : Config) (input : Nat → ℝ) (q : Nat) : ℝ :=
  VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.carry (value h input) c.BT (numChunks h c) q

def expected (h : Call) (c : Config) (input : Nat → ℝ) (t : Nat) : ℝ :=
  VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix (value h input) (numChunks h c*c.BT) t

structure Holds (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (input : Nat → ℝ) (remaining : Nat) (u : BlockState) : Prop where
  undef : u.undef=initial.undef
  count_le : remaining≤numChunks h c
  head : u.regs .nat [] "i_bh"=some (Tile.scalar row)
  carry : u.regs .real [] "b_z"=some (Tile.scalar (some (accumulated h c input remaining)))
  reads : ∀ t, t<h.T → u.readMem h.input (h.inputStrides.address h.H row t)=input t
  output : ∀ t, t<h.T → remaining*c.BT≤t →
    u.mem h.output (h.outputStrides.address h.H row t)=MemCell.real (expected h c input t)
  frame : BlockState.WriteWithin (RealMemory.footprint h row) initial u

private theorem fold_undef {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).undef=u.undef := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;>
      simp only [List.foldl_cons,hi,ite_true,ite_false,ih,BlockState.writeMem_undef]
    all_goals rfl

theorem iteration (initial : BlockState) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (row : Nat) (hr : row<h.B*h.H)
    (input : Nat → ℝ) (j : Int) (u : BlockState)
    (hj : j<1) (hjreg : u.regs .int [] "__rev_i_t"=some (Tile.scalar j))
    (inv : Holds initial h c row input ((-j).toNat+1) u) :
    ∃ v, stepStmts (loopBody h c) u=some v ∧ Holds initial h c row input (-j).toNat v := by
  let q := (-j).toNat
  let f := value h input
  let acc := accumulated h c input (q+1)
  let pre := RealStep.beforeStore h c row q f acc u
  let v := RealStep.result h c row q f acc u
  have inside : q<numChunks h c := by have := inv.count_le; omega
  have start := Geometry.chunk_start h c positive q inside
  have next : acc+chunkSum c.BT q f=accumulated h c input q :=
    VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.carry_predecessor f c.BT (numChunks h c) q
  have he := RealStep.executes h c row input acc j u hj start hjreg inv.head inv.carry inv.reads
  refine ⟨v,he,?_,?_,?_,?_,?_,?_,?_⟩
  · simp only [v,RealStep.result,RealMemory.scatter,fold_undef,RealStep.beforeStore,BlockState.setReg_undef]
    exact inv.undef
  · exact Nat.le_of_lt inside
  · simp [v,RealStep.result,RealMemory.scatter,RealStep.beforeStore,inv.head]
  · simp [v,RealStep.result,RealMemory.scatter,RealStep.beforeStore,next]
    rfl
  · intro t ht
    change (RealMemory.scatter h c row q f (acc+chunkSum c.BT q f) pre).readMem h.input _=_
    rw [RealMemory.preserves_input h layout c row q f _ pre]
    simpa [pre,RealStep.beforeStore] using inv.reads t ht
  · intro t ht lower
    change (RealMemory.scatter h c row q f (acc+chunkSum c.BT q f) pre).mem h.output _=_
    by_cases later : (q+1)*c.BT≤t
    · rw [RealMemory.preserves_later h layout c row q hr start f _ pre t ht later]
      simpa [pre,RealStep.beforeStore] using inv.output t ht later
    · have hi : t-q*c.BT<c.BT := by
        change q*c.BT≤t at lower
        rw [Nat.add_mul,Nat.one_mul] at later
        omega
      let i : Fin c.BT := ⟨t-q*c.BT,hi⟩
      have eq : q*c.BT+i.val=t := by dsimp [i]; change q*c.BT≤t at lower; omega
      have hit : q*c.BT+i.val<h.T := by simpa [eq] using ht
      have cell := RealMemory.cell h layout c row q hr start f (acc+chunkSum c.BT q f) pre i hit
      rw [eq] at cell
      rw [cell]
      apply congrArg MemCell.real
      rw [next]
      simpa only [eq] using VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.scalar_from_chunk f c.BT (numChunks h c) q i
  · have frame := RealMemory.frame h c row q start f (acc+chunkSum c.BT q f) pre
    intro r off hn
    exact (inv.frame r off hn).trans (frame r off hn)

abbrev initial := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit.initial
abbrev init_executes := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit.executes

theorem invariant (h : Call) (c : Config) (positive : 0<c.BT) (u : BlockState) :
    Holds u h c (u.pids 0) (fun t => u.readMem h.input (h.inputStrides.address h.H (u.pids 0) t))
      (numChunks h c) (initial u) := by
  refine ⟨rfl,le_rfl,?_,?_,?_,?_,?_⟩
  · simp [initial,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit.initial]
  · simp [initial,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealInit.initial,accumulated,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.carry_end]
  · intro t ht; rfl
  · intro t ht hl
    have hc : h.T≤numChunks h c*c.BT := Geometry.chunks_cover h c positive
    omega
  · intro r off hn; rfl

theorem mirror_invariant (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (input : Nat → ℝ) (q : Nat) (u : BlockState) (inv : Holds initial h c row input q u) (j : Int) :
    Holds initial h c row input q (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  refine ⟨inv.undef,inv.count_le,?_,?_,inv.reads,inv.output,inv.frame⟩
  · simpa using inv.head
  · simpa using inv.carry

theorem loop_iteration (initial : BlockState) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (row : Nat) (hr : row<h.B*h.H)
    (input : Nat → ℝ) (j : Int) (u : BlockState) (hj : j<1)
    (inv : Holds initial h c row input (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.remaining j) u) :
    ∃ v, stepStmts (loopBody h c) (u.setReg "__rev_i_t" .int [] (Tile.scalar j))=some v ∧
      Holds initial h c row input (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.remaining (j+1)) v := by
  have hm := mirror_invariant initial h c row input (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.remaining j) u inv j
  rw [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.current_chunk j hj] at hm
  obtain ⟨v,he,hv⟩ := iteration initial h layout c positive row hr input j
    (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) hj (by simp) hm
  exact ⟨v,he,by simpa only [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.after_step] using hv⟩

theorem statement_eval (h : Call) (c : Config) (u : BlockState) :
    stepStmt (statement h c) u=stepForRangeIntAux "__rev_i_t" (1-(numChunks h c:Int)) 1 1 (loopBody h c) u := by
  rw [statement,stepForRangeIntAux.forRangeInt_unfold,
    VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram.start_eval]
  simp only [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram.constInt_eval,
    evalOp_constNat,Option.bind_some,Option.bind_eq_bind,Tile.scalar_data]
  rfl

theorem loop_executes (initial : BlockState) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (row : Nat) (hr : row<h.B*h.H)
    (input : Nat → ℝ) (u : BlockState) (inv : Holds initial h c row input (numChunks h c) u) :
    ∃ v, stepStmt (statement h c) u=some v ∧ Holds initial h c row input 0 v := by
  obtain ⟨final,v,he,hf,hv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange.executes
    "__rev_i_t" 1 1 (loopBody h c) (fun j v => Holds initial h c row input (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.remaining j) v)
    (by decide) (fun j v hj hv => loop_iteration initial h layout c positive row hr input j v hj hv)
    (1-(numChunks h c:Int)) u (by simpa only [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.at_start] using inv)
  refine ⟨v,(statement_eval h c u).trans he,?_⟩
  simpa only [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.at_or_after_end final hf] using hv

theorem expected_suffix (h : Call) (c : Config) (positive : 0<c.BT) (input : Nat → ℝ) (t : Nat)
    (ht : t≤h.T) : expected h c input t=VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix input h.T t := by
  have eq := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.padding_contribution input (fun _ => 0) h.T (numChunks h c*c.BT) t
    (Geometry.chunks_cover h c positive) ht
  simpa [expected,value,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.padded,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.tail] using eq

def Post (h : Call) (initial final : BlockState) : Prop :=
  (∀ t, t<h.T → final.mem h.output (h.outputStrides.address h.H (initial.pids 0) t)=
    MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix (fun k => initial.readMem h.input
      (h.inputStrides.address h.H (initial.pids 0) k)) h.T t)) ∧
  BlockState.WriteWithin (RealMemory.footprint h (initial.pids 0)) initial final

theorem source_correct (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (u : BlockState) (hr : u.pids 0<h.B*h.H) :
    ∃ v, exec (Kernels.source h c).eraseDType.toAlgKernel u=some v ∧ Post h u v := by
  obtain ⟨v,he,inv⟩ := loop_executes u h layout c positive (u.pids 0) hr _ (initial u) (invariant h c positive u)
  refine ⟨v,?_,?_,inv.frame⟩
  · rw [exec,source_body,body,ProgramRewrite.append,init_executes]
    change stepStmts [statement h c] (initial u)=some v
    rw [stepStmts.cons_some he,stepStmts.nil]
  · intro t ht
    exact (inv.output t ht (by simp)).trans (congrArg MemCell.real (expected_suffix h c positive _ t (Nat.le_of_lt ht)))

theorem source_complete (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (u : BlockState) (hr : u.pids 0<h.B*h.H) :
    ∃ v, exec (Kernels.source h c).eraseDType.toAlgKernel u=some v ∧ Post h u v ∧ v.undef=u.undef := by
  obtain ⟨v,he,inv⟩ := loop_executes u h layout c positive (u.pids 0) hr _ (initial u) (invariant h c positive u)
  refine ⟨v,?_,⟨?_,inv.frame⟩,inv.undef⟩
  · rw [exec,source_body,body,ProgramRewrite.append,init_executes]
    change stepStmts [statement h c] (initial u)=some v
    rw [stepStmts.cons_some he,stepStmts.nil]
  · intro t ht
    exact (inv.output t ht (by simp)).trans (congrArg MemCell.real (expected_suffix h c positive _ t (Nat.le_of_lt ht)))

#print axioms iteration
#print axioms loop_executes
#print axioms source_correct
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealExecution
