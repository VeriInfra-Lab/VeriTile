import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.AllocatorWitness
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.FP

/-! A positive-size complete same-call inhabitant of the strengthened boundary.
The two inputs alias allocation 0; actual outputs are allocations 1 and 2.
Every allocation has three trailing bytes beyond its one accessed FP32 word. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.WholeWitness
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd RawBoundary HostModel HostStorage
set_option autoImplicit false
noncomputable section

def entry : PhysicalMemory Word := fun _ _ => 2
def old (a : Nat) : Prop := a = 0
def call : Call Word :=
  ⟨⟨[1,1,1],fun _ => 1,⟨0,0⟩,0⟩,⟨[1],fun _ => 1,⟨0,0⟩,0⟩,none,1,Bool.false,0,0x7f800000⟩
def geometry : HostModel.Geometry call where
  host := makeHost call 1 1 1 (by decide)
  shape := rfl
  chunk := rfl
  biasShape := by intro t h; cases h
  rateShape := rfl
def allocated : AllocationResult Word call geometry.host entry old where
  out := ⟨1,0⟩
  sum := ⟨2,0⟩
  distinct := by decide
  memory := entry
  dtOld := rfl
  rateOld := rfl
  biasOld := rfl
  outFresh := by norm_num [old]
  sumFresh := by norm_num [old]
  frame := by intros; rfl
  outObject := 20
  sumObject := 21
  objectsDistinct := by decide
def logical : Logical Word := ⟨fun _ _ _ => 2,fun _ => 2,fun _ => 2⟩
def prepared : Prepared call geometry logical allocated where
  dt := call.dt
  rate := call.rate
  bias := none
  memory := entry
  dtShape := rfl
  rateShape := rfl
  biasPresent := rfl
  biasShape := by intro t h; cases h
  dtDevice := rfl
  rateDevice := rfl
  biasDevice := by intro t h; cases h
  kept := old
  callerProtected := by intro a h; exact h
  dtProtected := rfl
  rateProtected := rfl
  biasProtected := rfl
  outFresh := by norm_num [old,allocated]
  sumFresh := by norm_num [old,allocated]
  frame := by intros; rfl
  dtValues := by intros; rfl
  rateValues := by intros; rfl
  biasValues := by intro t h; cases h
  dtOriginal := rfl
  rateOriginal := rfl
  biasOriginal := rfl
  memoryOriginal := rfl


def backings : Backings prepared where
  allocation := AllocatorWitness.allocation
  live := AllocatorWitness.live
  keptLive := by intro a h; change a = 0 at h; subst a; norm_num [AllocatorWitness.live]
  separated := AllocatorWitness.separated
  dt := { AllocatorWitness.tensor 0 10 with shape := [1,1,1] }
  rate := AllocatorWitness.tensor 0 11
  bias := AllocatorWitness.tensor 0 12
  out := { AllocatorWitness.tensor 1 20 with shape := [1,1,1,1] }
  sum := { AllocatorWitness.tensor 2 21 with shape := [1,1,1,1] }
  dtView := rfl
  rateView := rfl
  biasView := rfl
  outView := rfl
  sumView := rfl
  dtShape := rfl
  rateShape := rfl
  dtStride := rfl
  rateStride := rfl
  dtDevice := rfl
  rateDevice := rfl
  biasDevice := by intro t h; cases h
  biasMetadata := by intro t h; cases h
  outShape := by decide
  sumShape := by decide
  outPacked := by decide
  sumPacked := by decide
  outObject := rfl
  sumObject := rfl
  outDevice := rfl
  sumDevice := rfl
  dtLive := by norm_num [AllocatorWitness.live,AllocatorWitness.tensor]
  rateLive := by norm_num [AllocatorWitness.live,AllocatorWitness.tensor]
  biasLive := by intros; norm_num [AllocatorWitness.live,AllocatorWitness.tensor]
  outLive := by norm_num [AllocatorWitness.live,AllocatorWitness.tensor]
  sumLive := by norm_num [AllocatorWitness.live,AllocatorWitness.tensor]
  dtCovers := by
    intro b t h hb ht hh
    have bz : b = 0 := by change b < 1 at hb; omega
    have tz : t = 0 := by change t < 1 at ht; omega
    have hz : h = 0 := by change h < 1 at hh; omega
    subst b; subst t; subst h; norm_num [Storage.Tensor.Covers,AllocatorWitness.tensor,AllocatorWitness.allocation]
  rateCovers := by
    intro h hh
    have hz : h = 0 := by change h < 1 at hh; omega
    subst h; norm_num [Storage.Tensor.Covers,AllocatorWitness.tensor,AllocatorWitness.allocation]
  biasCovers := by intro impossible; cases impossible
  outCovers := by
    intro off h
    have hz : off = 0 := by change off < 1 at h; omega
    subst off; norm_num [Storage.Tensor.Covers,AllocatorWitness.tensor,AllocatorWitness.allocation]
  sumCovers := by
    intro off h
    have hz : off = 0 := by change off < 1 at h; omega
    subst off; norm_num [Storage.Tensor.Covers,AllocatorWitness.tensor,AllocatorWitness.allocation]

theorem whole_boundary_inhabited : Nonempty (Backings prepared) := ⟨backings⟩

def singletonPlan (heads : Nat) : Backend.PrefixPlan heads 1 where
  tree := fun _ j => .leaf j
  leaves := by
    intro i j
    have hj : j.val = 0 := by omega
    simp [Backend.SumTree.leaves,hj]

def plan (model : RawObserved.NativeModel Word) : LaunchPlan Word prepared.host where
  block := ⟨1,by simp⟩
  model := model
  scan := by
    have eq : prepared.host.BC = 1 := by decide
    rw [eq]
    exact singletonPlan 1
  order := (reblock prepared.host ⟨1,by simp⟩).grid
  complete := List.Perm.refl _

/-- A positive-size source trial exists for every total scalar model, with
the strengthened flat-byte observations. No output law is assumed. -/
theorem positive_source_execution (model : RawObserved.NativeModel Word) :
    ∃ final, runTrials prepared.views prepared.host call.hasBias call.softplus call.lower call.upper
      [plan model] prepared.memory = some final ∧ FPProof.FlatObservables backings final :=
  FPProof.trials_flat backings [plan model]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.WholeWitness
