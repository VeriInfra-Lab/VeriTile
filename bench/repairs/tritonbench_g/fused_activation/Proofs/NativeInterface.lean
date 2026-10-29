import bench.repairs.tritonbench_g.fused_activation.IO
import bench.repairs.tritonbench_g.fused_activation.Proofs.AccessSafety
import bench.repairs.tritonbench_g.fused_activation.Proofs.NativeRewrite

namespace VeriTile.Bench.Repairs.FusedActivation.NativeInterface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
noncomputable section

/-- The selected native lowering retains its FMA and instruction-sized sigmoid
ports. This theorem executes the same two kernels and proves equality of all
typed native memory observations, plus the exact active-store exterior frame.
The existing finite domain is preserved; addition commutation follows from
the selected concrete FP32 model. -/
theorem execution_equiv (rest : FP.Structural.Algebra Graph.Expr)
    (ops : Graph.Operations) (cfg : FP.Config) (binding : NativeRewrite.AddBinding ops cfg)
    (D : FP.Guarded.Domain Graph.Word)
    (p : Parameters) (W : Nat) (a : Bool) (s : FP.Structural.State Graph.Expr)
    (xs biases inputs : Nat → Graph.Word)
    (hx : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.output (offset p s j)).read .real =
        .leaf (xs (offset p s j)))
    (hb : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.y (offset p s j % W)).read .real =
        .leaf (biases (offset p s j % W)))
    (hi : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.x (offset p s j)).read .real =
        .leaf (inputs (offset p s j)))
    (domain : ∀ j : TileIndex [p.block], active p s j →
      D .finite (ops.fmaRN32
        (inputs (offset p s j)) Graph.half
        (xs (offset p s j))) ∧
      D .finite (biases (offset p s j % W))) :
    ∃ first second,
      FP.Structural.exec (Graph.algebra rest) (candidateKernel p W (1/2) a) s = some first ∧
      FP.Structural.exec (Graph.algebra rest)
        (commutedKernel p.output p.y p.x W p.n p.block (1/2) a) s = some second ∧
      NativeRewrite.observeMemory ops first.mem = NativeRewrite.observeMemory ops second.mem ∧
      (∀ r o, (r ≠ p.output ∨ ∀ j, active p s j →
        o ≠ offset p s j) →
        first.mem r o = s.mem r o ∧ second.mem r o = s.mem r o) := by
  refine ⟨(execution (Graph.algebra rest) p W (1/2) a s).val,
    (Rewrite.commutedExecution (Graph.algebra rest) p W (1/2) a s).val,
    (execution (Graph.algebra rest) p W (1/2) a s).property,
    (Rewrite.commutedExecution (Graph.algebra rest) p W (1/2) a s).property,
    NativeRewrite.memory_equiv rest ops cfg binding D p W a s xs biases inputs hx hb hi domain, ?_⟩
  intro r o outside
  constructor
  · rw [execution_memory]
    exact expected_frame (Graph.algebra rest) p W (1/2) a s r o outside
  · exact Rewrite.commuted_frame (Graph.algebra rest) p W (1/2) a s r o outside

@[simp] theorem observeCell_eq (ops : Graph.Operations) (cell : Cell Graph.Expr) :
    mapCell (Graph.eval ops) cell = NativeRewrite.observeCell ops cell := by
  cases cell with
  | mk d v => cases d <;> rfl

private def initial (launch : Nat → Nat) (s : State Graph.Expr) : State Graph.Expr :=
  { s with pids := launch }

private theorem input_binding (rest : Algebra Graph.Expr) (ops : Graph.Operations)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat)
    (values : NativeIO.Inputs p) (s : State Graph.Expr)
    (hin : (NativeIO.original rest ops p W a launch).InputBinding values s) :
    (∀ j : Fin p.block, launch 0*p.block+j.val<p.n →
      (s.mem p.output (launch 0*p.block+j.val)).read .real = .leaf (NativeIO.x values j)) ∧
    (∀ j : Fin p.block, launch 0*p.block+j.val<p.n →
      (s.mem p.y ((launch 0*p.block+j.val)%W)).read .real = .leaf (NativeIO.bias values j)) ∧
    (∀ j : Fin p.block, launch 0*p.block+j.val<p.n →
      (s.mem p.x (launch 0*p.block+j.val)).read .real = .leaf (NativeIO.input values j)) := by
  exact ⟨fun j hj => hin ⟨0, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj,
    fun j hj => hin ⟨1, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj,
    fun j hj => hin ⟨2, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj⟩

private theorem safety (rest : Algebra Graph.Expr) (ops : Graph.Operations)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat)
    (values : NativeIO.Inputs p) (bounds : RegionBounds) (s : State Graph.Expr)
    (hb : (NativeIO.original rest ops p W a launch).InBounds bounds values) :
    ScheduleSafe bounds ((NativeIO.original rest ops p W a launch).schedule values) s ∧
    ScheduleSafe bounds ((NativeIO.commuted rest ops p W a launch).schedule values) s := by
  simp only [NativeIO.original, NativeIO.commuted, scheduleSafe_singleton]
  change TraceSafe (Graph.algebra rest) bounds
      (candidateKernel p W (1/2) a) (initial launch s) ∧
    TraceSafe (Graph.algebra rest) bounds
      (commutedKernel p.output p.y p.x W p.n p.block (1/2) a) (initial launch s)
  exact AccessSafety.structural_traceSafe (Graph.algebra rest) p.output p.y p.x W p.n p.block
    (1/2) a bounds (initial launch s)
    (fun j hj => hb.2.1 ⟨0, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj)
    (fun j hj => hb.2.1 ⟨1, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj)
    (fun j hj => hb.2.1 ⟨2, by simp [NativeIO.original, NativeIO.memory, NativeIO.inputs]⟩ (j,PUnit.unit) hj)

/-- Bind logical word channels to the existing native pair proof. The exact
source-safety certificates are composed independently of the rewrite law. -/
theorem equivalent (rest : Algebra Graph.Expr)
    (ops : Graph.Operations) (cfg : FP.Config) (binding : NativeRewrite.AddBinding ops cfg)
    (D : FP.Guarded.Domain Graph.Word)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat) :
    (NativeIO.original rest ops p W a launch).EquivalentOn
      (NativeIO.commuted rest ops p W a launch) (NativeIO.domain ops D p launch) := by
  apply ProgramIO.EquivalentOn.intro
  · intro values _ bounds hb s _
    exact (safety rest ops p W a launch values bounds s hb).1
  · intro values _ bounds hb s _
    exact (safety rest ops p W a launch values bounds s hb).2
  · intro values hd bounds hb s hin
    obtain ⟨hx,hbias,hi⟩ := input_binding rest ops p W a launch values s hin
    let xs := fun o => Graph.eval ops ((s.mem p.output o).read .real)
    let biases := fun o => Graph.eval ops ((s.mem p.y o).read .real)
    let inputs := fun o => Graph.eval ops ((s.mem p.x o).read .real)
    obtain ⟨first,second,he,hc,equal,frame⟩ := execution_equiv rest ops cfg binding D
      p W a (initial launch s) xs biases inputs
      (by intro j active; change _ = Graph.Expr.leaf (Graph.eval ops _); dsimp only [initial,offset]; rw [hx j.1 active]; rfl)
      (by intro j active; change _ = Graph.Expr.leaf (Graph.eval ops _); dsimp only [initial,offset]; rw [hbias j.1 active]; rfl)
      (by intro j active; change _ = Graph.Expr.leaf (Graph.eval ops _); dsimp only [initial,offset]; rw [hi j.1 active]; rfl)
      (by
        intro j active
        simpa only [xs,biases,inputs,initial,offset,hi j.1 active,hx j.1 active,hbias j.1 active,Graph.eval] using hd j.1 active)
    refine ⟨first,second,?_,?_,?_,?_,?_⟩
    · simpa only [NativeIO.original,runSchedule_singleton] using he
    · simpa only [NativeIO.commuted,runSchedule_singleton] using hc
    · intro o j active
      fin_cases o
      simpa only [observeCell_eq] using congrFun (congrFun equal p.output)
        (launch 0*p.block+j.1.val)
    · intro r o outside
      apply (frame r o ?_).1
      by_cases hr : r=p.output
      · right
        intro j active
        exact (outside.1 ⟨0,by simp [NativeIO.original,NativeIO.memory,NativeIO.outputs]⟩ j active).resolve_left (not_ne_iff.mpr hr)
      · exact Or.inl hr
    · intro r o outside
      apply (frame r o ?_).2
      by_cases hr : r=p.output
      · right
        intro j active
        exact (outside.1 ⟨0,by simp [NativeIO.original,NativeIO.memory,NativeIO.outputs]⟩ j active).resolve_left (not_ne_iff.mpr hr)
      · exact Or.inl hr

/-- Exact source identity for the profiled program used by the public FP entry. -/
theorem original_source (W N B : Nat) (a : Bool) :
    ({ kernel := candidateKernel ⟨"input", "bias", "x", N, B⟩ W (1/2) a,
       domain := [Rewrite.additionSite ⟨"input", "bias", "x", N, B⟩ W (1/2) a] } :
      FP.ProfiledMemory.Program).kernel = kernel "x" "bias" "input" W N B (1/2) a := rfl

#axiomsClean original_source
#axiomsClean execution_equiv
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.FusedActivation.NativeInterface
