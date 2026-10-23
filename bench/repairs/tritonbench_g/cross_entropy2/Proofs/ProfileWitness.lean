import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Floating

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- The actual lane-zero tree of the two pinned BLOCK_SIZE=4 forward
profiles: butterfly shuffle offsets 2 then 1. The evidence checker verifies
both dataflow edges and the reducer opcode in the saved PTX. -/
def sm90Four : Plan 4 :=
  ⟨.node (.node (.leaf 0) (.leaf 2)) (.node (.leaf 1) (.leaf 3)),by decide,by decide⟩

def butterfly {α : Type} (op : α → α → α) (offset : Nat) (input : Nat → α) : Nat → α :=
  fun lane => op (input lane) (input (Nat.xor lane offset))

theorem sm90_four_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) :
    butterfly op 1 (butterfly op 2 input) 0=sm90Four.tree.run op seed input := rfl

/-- Four independent seeded accumulators, two active input slots, then the
literal left-to-right accumulator combine loop in installed Reduce.cuh.
This is a local loop witness; binding Torch dispatcher/template selection
to this path remains separate from this theorem. -/
def threadTwoFour : Plan 2 :=
  ⟨.node (.node (.node (.node .seed (.leaf 0)) (.node .seed (.leaf 1))) .seed) .seed,
    by decide,by decide⟩

theorem thread_two_four_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) :
    op (op (op (op seed (input 0)) (op seed (input 1))) seed) seed=
      threadTwoFour.tree.run op seed input := rfl

def seededRange : Nat → Tree
  | 0 => .seed
  | n+1 => .node (seededRange n) (.leaf n)

theorem seeded_range_leaves (n : Nat) : (seededRange n).leaves=List.range n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [seededRange,Tree.leaves,ih,List.range_succ]

def witnessPlan (width : Nat) (positive : 0<width) : Plan width :=
  ⟨seededRange width,(seeded_range_leaves width) ▸ List.Perm.refl _,positive⟩

/-- A total family of lawful nonempty plans exists. The finite native
witness overrides the exact four-lane device tree and two-slot host loop;
other widths here are merely formal witnesses, not native bindings. -/
def witnessProfile : Profile where
  sumTree := fun width positive => if h : width=4 then h ▸ sm90Four else witnessPlan width positive
  maxTree := fun width positive => if h : width=4 then h ▸ sm90Four else witnessPlan width positive
  hostSumTree := fun width positive => if h : width=2 then h ▸ threadTwoFour else witnessPlan width positive
  hostMaxTree := fun width positive => if h : width=2 then h ▸ threadTwoFour else witnessPlan width positive

theorem ports_reflexive (n : Ports) : Obligations n n := by
  constructor <;> intros <;> rfl

def sampleOptions : Host.Options :=
  ⟨1/4,1,1/8,-100,8,0,4,Bool.true,Bool.false⟩
def sampleLogits : Tensor :=
  ⟨0,"logits",[32,8],[8,1],0,Bool.false,Bool.true,0,"torch.float32"⟩
def sampleLabels : Tensor :=
  ⟨1,"labels",[32],[1],0,Bool.false,Bool.true,0,"torch.int32"⟩

theorem metadata_nonempty : Host.valid sampleOptions sampleLogits sampleLabels 32 8 := by
  norm_num [Host.valid,sampleOptions,sampleLogits,sampleLabels]
  exact ⟨2,by decide⟩

#axiomsClean sm90_four_exact
#axiomsClean thread_two_four_exact
#axiomsClean seeded_range_leaves
#axiomsClean ports_reflexive
#axiomsClean metadata_nonempty
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
