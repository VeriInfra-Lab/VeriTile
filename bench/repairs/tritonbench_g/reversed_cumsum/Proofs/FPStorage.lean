import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPReference
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPStorage
open VeriTile.Triton FP.Structural FPValues Geometry
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

abbrev Heap (α : Type) := Nat → Cell α

def entry (h : Call) (p : Storage.Placement h) (program : Program h) (heap : Heap α) : State α where
  mem := Storage.lift h p (.mk .real default) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then program.2.val else if ax=1 then program.1.val else 0
  numPids := fun ax => if ax=0 then featureChunks h else if ax=1 then h.B*h.H else 1
  undef := fun d _ _ => defaultValue d

def execute (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (program : Program h) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (machine M dot) (Kernels.source h c) (entry h p program heap)).map
    (fun v => Storage.commit h p heap v.mem)

def input (h : Call) (p : Storage.Placement h) (heap : Heap α) (row t s : Nat) : α :=
  toFloat h.inputDType ((heap (Storage.address p.backing.input (Geometry.address h row t s))).read h.inputDType.toTileDType)

def localExpected (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (heap : Heap α) (row f t : Nat) (j : Fin BS) : Cell α :=
  FPMemory.stored M h (FPExecution.expected M dot h c positive f (input h p heap row) t j)

def expected (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (heap : Heap α) (row t s : Nat) : Cell α :=
  localExpected M dot h p c positive heap row (s/BS) t ⟨s%BS,Nat.mod_lt _ (by decide)⟩

theorem local_global (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (heap : Heap α) (row f t : Nat) (j : Fin BS) :
    localExpected M dot h p c positive heap row f t j=expected M dot h p c positive heap row t (f*BS+j.val) := by
  unfold expected
  have div : (f*BS+j.val)/BS=f := by have := j.isLt; dsimp [BS] at *; omega
  have mod : (f*BS+j.val)%BS=j.val := by have := j.isLt; dsimp [BS] at *; omega
  rw [div]
  congr 1
  exact Fin.ext mod.symm

theorem complete (M : Algebra α) (dot : DotBackend α) (h : Call) (distinct : h.input≠h.output)
    (p : Storage.Placement h) (c : Config) (positive : 0<c.BT) (program : Program h) (heap : Heap α) :
    ∃ final, execute M dot h p c program heap=some final ∧
      (∀ off, off<p.backing.input.extent → final (Storage.address p.backing.input off)=heap (Storage.address p.backing.input off)) ∧
      (∀ t, ∀ j : Fin BS, t<h.T → program.2.val*BS+j.val<h.S →
        final (Storage.address p.backing.output (Geometry.address h program.1.val t (program.2.val*BS+j.val)))=
          localExpected M dot h p c positive heap program.1.val program.2.val t j) ∧
      (∀ addr, ¬Storage.writes h p [program] addr → final addr=heap addr) := by
  obtain ⟨v,he,out,frame⟩ := FPExecution.source_correct M dot h distinct c positive program.1 program.2
    (entry h p program heap) (by simp [entry]) (by simp [entry])
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => ReversedCumsumScalar.Storage.commit_input h.flat p heap v.mem off inside
  · intro t j ht hs
    dsimp only [Storage.commit,Storage.address]
    rw [ReversedCumsumScalar.Storage.commit_output h.flat p heap v.mem _ (Storage.window h p program.1 t _ ht hs).2]
    erw [out t j ht hs]
    unfold localExpected
    apply congrArg (FPMemory.stored M h)
    apply FPReference.expected_congr
    intro t s ht hs
    simp [FPExecution.input,entry,Storage.lift,ReversedCumsumScalar.Storage.lift,Call.flat,input,Storage.address]
  · intro addr outside
    unfold Storage.commit ReversedCumsumScalar.Storage.commit
    cases eq : ReversedCumsumScalar.Storage.decode p.backing.output addr with
    | none => rfl
    | some off =>
      have spec := ReversedCumsumScalar.Storage.decode_some _ _ _ eq
      have no : ¬FPMemory.footprint h program.1.val program.2.val (h.output,off) := by
        rintro ⟨_,t,j,ht,hs,address⟩
        dsimp only at address
        exact outside ⟨program,by simp,t,j,ht,hs,by rw [address] at spec; exact spec.2⟩
      change v.mem h.output off=heap addr
      rw [←frame h.output off no]
      change ReversedCumsumScalar.Storage.lift h.flat p (.mk .real default) heap h.flat.output off=heap addr
      rw [ReversedCumsumScalar.Storage.lift_output h.flat (flat_layout h distinct) p,←spec.2]

#print axioms complete
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPStorage
