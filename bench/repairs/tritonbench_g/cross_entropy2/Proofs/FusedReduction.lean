import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AlignedProfile
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Lowering

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 10000000

/-- A reduction instruction graph that distinguishes rounded products,
unrounded FMA operands and a shared select. No FMA/add identification occurs. -/
inductive SmoothingProgram where
  | zero
  | product (index : Nat)
  | scaled (index : Nat)
  | fused (index : Nat) (accumulator : SmoothingProgram)
  | masked (index : Nat) (body : SmoothingProgram)
  | add (left right : SmoothingProgram)

def SmoothingProgram.leaves : SmoothingProgram → List Nat
  | .zero => []
  | .product i | .scaled i => [i]
  | .fused i acc => i::acc.leaves
  | .masked _ body => body.leaves
  | .add a b => a.leaves++b.leaves

def SmoothingProgram.run (n : Ports) (scale : Word) (input : Nat → Word)
    (active : Nat → Bool) : SmoothingProgram → Word
  | .zero => n.literal 0
  | .product i => n.mul scale (input i)
  | .scaled i => if active i then n.mul scale (input i) else n.literal 0
  | .fused i acc => n.fma scale (input i) (acc.run n scale input active)
  | .masked i body => if active i then body.run n scale input active else n.literal 0
  | .add a b => n.add (a.run n scale input active) (b.run n scale input active)

def SmoothingProgram.emit (scale : Instruction) (input : Nat → Instruction)
    (active : Nat → Bool) : SmoothingProgram → Instruction
  | .zero => .literal 0
  | .product i => .binary .mul scale (input i)
  | .scaled i => .select (active i) (.binary .mul scale (input i)) (.literal 0)
  | .fused i acc => .fma scale (input i) (acc.emit scale input active)
  | .masked i body => .select (active i) (body.emit scale input active) (.literal 0)
  | .add a b => .binary .add (a.emit scale input active) (b.emit scale input active)

theorem smoothing_emit_exact (n : Ports) (profile : Profile) (scale : Instruction)
    (input : Nat → Instruction) (active : Nat → Bool) (program : SmoothingProgram) :
    runInstruction n profile (program.emit scale input active)=
      program.run n (runInstruction n profile scale) (fun i => runInstruction n profile (input i)) active := by
  induction program with
  | zero | product i | scaled i => rfl
  | fused i acc ih => simp only [SmoothingProgram.emit,runInstruction,SmoothingProgram.run,ih]
  | masked i body ih => simp only [SmoothingProgram.emit,runInstruction,SmoothingProgram.run,ih]
  | add a b ia ib => simp only [SmoothingProgram.emit,runInstruction,SmoothingProgram.run,ia,ib]

/-- Exact first-four register graph in the aligned SM90 trace. x1 remains
rounded; x0, x2 and x3 enter three FMAs, followed by one shared select. -/
def alignedPrefix (lane : Nat) : SmoothingProgram :=
  .masked (alignedCoordinate lane 0)
    (.fused (alignedCoordinate lane 3)
      (.fused (alignedCoordinate lane 2)
        (.fused (alignedCoordinate lane 0) (.product (alignedCoordinate lane 1)))))

/-- Remaining registers put the new rounded masked product first. -/
def alignedSmoothingChain (lane : Nat) : Nat → SmoothingProgram
  | 0 => alignedPrefix lane
  | n+1 => .add (.scaled (alignedCoordinate lane (n+4))) (alignedSmoothingChain lane n)

def alignedSmoothing : SmoothingProgram :=
  lane128.tree.run SmoothingProgram.add .zero (fun lane => alignedSmoothingChain lane 508)

theorem aligned_prefix_exact (n : Ports) (scale : Word) (input : Nat → Word)
    (active : Nat → Bool) (lane : Nat) :
    (alignedPrefix lane).run n scale input active=
      if active (4*lane) then
        n.fma scale (input (4*lane+3))
          (n.fma scale (input (4*lane+2))
            (n.fma scale (input (4*lane)) (n.mul scale (input (4*lane+1)))))
      else n.literal 0 := by
  simp [alignedPrefix,alignedCoordinate,SmoothingProgram.run]

theorem aligned_smoothing_chain_covers (lane n : Nat) :
    (alignedSmoothingChain lane n).leaves.Perm ((List.range (n+4)).map (alignedCoordinate lane)) := by
  induction n with
  | zero =>
    have permutation : ([3,2,0,1] : List Nat).Perm (List.range 4) := by decide
    exact permutation.map (alignedCoordinate lane)
  | succ n ih =>
    change ([alignedCoordinate lane (n+4)]++(alignedSmoothingChain lane n).leaves).Perm _
    rw [show n+1+4=(n+4)+1 by omega,List.range_succ,List.map_append,List.map_cons,List.map_nil]
    exact (List.Perm.append_left _ ih).trans (List.perm_append_comm)

theorem smoothing_tree_leaves (tree : Tree) (inputs : Nat → SmoothingProgram) :
    (tree.run SmoothingProgram.add .zero inputs).leaves=tree.leaves.flatMap (fun i => (inputs i).leaves) := by
  induction tree with
  | leaf i => simp [Tree.run,Tree.leaves]
  | seed => rfl
  | node a b ia ib => simp only [Tree.run,SmoothingProgram.leaves,Tree.leaves,ia,ib,List.flatMap_append]

theorem aligned_smoothing_covers : alignedSmoothing.leaves.Perm (List.range 65536) := by
  rw [alignedSmoothing,smoothing_tree_leaves]
  apply List.Perm.trans (lane128.covers.flatMap_right _)
  apply List.Perm.trans _ aligned_range
  apply List.Perm.flatMap_left
  intro lane member
  exact aligned_smoothing_chain_covers lane 508

theorem smoothing_tree_run (n : Ports) (scale : Word) (input : Nat → Word)
    (active : Nat → Bool) (tree : Tree) (programs : Nat → SmoothingProgram) :
    (tree.run SmoothingProgram.add .zero programs).run n scale input active=
      tree.run n.add (n.literal 0) (fun i => (programs i).run n scale input active) := by
  induction tree with
  | leaf i => rfl
  | seed => rfl
  | node a b ia ib => simp only [Tree.run,SmoothingProgram.run,ia,ib]

/-- Exact register/warp/shared composition, without reassociation. -/
theorem aligned_smoothing_exact (n : Ports) (scale : Word) (input : Nat → Word) (active : Nat → Bool) :
    alignedSmoothing.run n scale input active=
      sm90Four.tree.run n.add (n.literal 0) (fun warp =>
        (warpTree warp 0).run n.add (n.literal 0) (fun lane =>
          (alignedSmoothingChain lane 508).run n scale input active)) := by
  rw [alignedSmoothing,smoothing_tree_run]
  change (substitute (fun warp => warpTree warp 0) sm90Four.tree).run _ _ _=_
  rw [substitute_run]

/-- Divisibility metadata justifies one guard for each group of four.
The width and block origin are multiples of four; partial final groups cannot
occur for this profile. Other metadata select other profiles. -/
theorem aligned_shared_mask (cols base lane slot : Nat) (colsAligned : cols%4=0) (baseAligned : base%4=0) :
    (base+alignedCoordinate lane slot<cols) ↔
      (base+4*lane+512*(slot/4)<cols) := by
  unfold alignedCoordinate
  omega

#axiomsClean smoothing_emit_exact
#axiomsClean aligned_prefix_exact
#axiomsClean aligned_smoothing_covers
#axiomsClean aligned_smoothing_exact
#axiomsClean aligned_shared_mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
