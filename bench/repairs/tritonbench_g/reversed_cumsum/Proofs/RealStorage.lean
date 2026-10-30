import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealExecution
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealStorage
open VeriTile.Triton Geometry
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

abbrev Heap := Nat → MemCell

def read (cell : MemCell) : ℝ := match cell.readAs .real with
  | some (some value) => value
  | _ => 0

def entry (h : Call) (p : Storage.Placement h) (program : Program h) (heap : Heap) : BlockState where
  mem := Storage.lift h p (MemCell.real 0) heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then program.2.val else if ax=1 then program.1.val else 0
  numPids := fun ax => if ax=0 then featureChunks h else if ax=1 then h.B*h.H else 1
  undef := fun _ _ => 0

def execute (h : Call) (p : Storage.Placement h) (c : Config) (program : Program h) (heap : Heap) : Option Heap :=
  (exec (Kernels.source h c).eraseDType.toAlgKernel (entry h p program heap)).map
    (fun v => Storage.commit h p heap v.mem)

def input (h : Call) (p : Storage.Placement h) (heap : Heap) (row t s : Nat) : ℝ :=
  read (heap (Storage.address p.backing.input (Geometry.address h row t s)))

def expected (h : Call) (p : Storage.Placement h) (heap : Heap) (row t s : Nat) : MemCell :=
  MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffixValue (fun k => input h p heap row k s) h.T t)

abbrev window := Storage.window

theorem complete (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (program : Program h) (heap : Heap) :
    ∃ final, execute h p c program heap=some final ∧
      (∀ off, off<p.backing.input.extent → final (Storage.address p.backing.input off)=heap (Storage.address p.backing.input off)) ∧
      (∀ t, ∀ j : Fin BS, t<h.T → program.2.val*BS+j.val<h.S →
        final (Storage.address p.backing.output (Geometry.address h program.1.val t (program.2.val*BS+j.val)))=
          expected h p heap program.1.val t (program.2.val*BS+j.val)) ∧
      (∀ addr, ¬Storage.writes h p [program] addr → final addr=heap addr) := by
  obtain ⟨v,he,out,frame⟩ := RealExecution.source_correct h c program.1.val program.2
    (entry h p program heap) distinct positive (by simp [entry]) (by simp [entry])
  refine ⟨Storage.commit h p heap v.mem,?_,?_,?_,?_⟩
  · simp [execute,he]
  · exact fun off inside => ReversedCumsumScalar.Storage.commit_input h.flat p heap v.mem off inside
  · intro t j ht hs
    dsimp only [Storage.commit,Storage.address]
    erw [ReversedCumsumScalar.Storage.commit_output h.flat p heap v.mem _ (window h p program.1 t _ ht hs).2]
    erw [out t j ht hs]
    simp [expected,entry,RealExecution.suffix,BlockState.readMem,input,read,Storage.lift,ReversedCumsumScalar.Storage.lift,Call.flat,Storage.address]
    rfl
  · intro addr outside
    unfold Storage.commit ReversedCumsumScalar.Storage.commit
    cases eq : ReversedCumsumScalar.Storage.decode p.backing.output addr with
    | none => rfl
    | some off =>
      have spec := ReversedCumsumScalar.Storage.decode_some _ _ _ eq
      have no : ¬RealExecution.footprint h program.1.val program.2.val (h.output,off) := by
        rintro ⟨_,t,j,⟨ht,hs⟩,address⟩
        have addrEq : off=Geometry.address h program.1.val t (program.2.val*BS+j.val) := by
          simpa [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,RealProgram.layout,RealProgram.launch,Geometry.address] using address
        exact outside ⟨program,by simp,t,j,ht,hs,by rw [addrEq] at spec; exact spec.2⟩
      change v.mem h.output off=heap addr
      rw [←frame h.output off no]
      change Storage.lift h p (MemCell.real 0) heap h.output off=heap addr
      dsimp only [Storage.lift]
      change ReversedCumsumScalar.Storage.lift h.flat p (MemCell.real 0) heap h.flat.output off=heap addr
      rw [ReversedCumsumScalar.Storage.lift_output h.flat (flat_layout h distinct) p,←spec.2]

#print axioms complete
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealStorage
