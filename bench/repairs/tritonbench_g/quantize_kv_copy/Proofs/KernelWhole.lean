import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelRaw
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelModelResult
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Contract

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelWhole
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000

def parameters (m : Inputs) (words : Nat → MetadataWord) : Parameters :=
  ⟨(request m words).groups,groupBlock (request m words),(request m words).headCount⟩

/-- Input-only logical reads; arbitrary Real inputs are never restricted to a
finite FP32 codec. Flattened program coordinates denote token * heads + head. -/
structure Reads {α : Type} [Inhabited α] (m : Inputs) (words : Nat → MetadataWord)
    (values : Nat → Nat → Nat → α) (s : State α) : Prop where
  input : ∀ pid g c,pid<(request m words).programs → g<(request m words).groups → c<8 →
    (s.mem "K" (inputOffset (parameters m words) pid g c)).read .real=values pid g c
  destination : ∀ token,token<(request m words).tokens →
    (s.mem "D" token).read .int=(words token).toInt

/-- Both caller outputs and the complete typed-memory frame. The target has no
execution, source registers or store-event dependency. -/
def Result {α : Type} (m : Inputs) (words : Nat → MetadataWord)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (initial final : State α) : Prop :=
  (∀ pid g c,pid<(request m words).programs → g<(request m words).groups → c<8 →
    final.mem "O" (codeOffset (parameters m words) ((request m words).destination pid) g c)=.mk .int (codes pid g c)) ∧
  (∀ pid g,pid<(request m words).programs → g<(request m words).groups →
    final.mem "S" (scaleOffset (parameters m words) ((request m words).destination pid) g)=.mk .real (scales pid g)) ∧
  (∀ (r : RegionName) o,
    (r≠"O" ∨ ∀ pid g c,pid<(request m words).programs → g<(request m words).groups → c<8 →
      o≠codeOffset (parameters m words) ((request m words).destination pid) g c) →
    (r≠"S" ∨ ∀ pid g,pid<(request m words).programs → g<(request m words).groups →
      o≠scaleOffset (parameters m words) ((request m words).destination pid) g) →
    final.mem r o=initial.mem r o)

 theorem whole {α : Type} [Inhabited α] (op : Ops α) (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (initial : State α)
    (metadata : ∀ token,token<(request m words).tokens → (initial.mem "D" token).read .int=(words token).toInt)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (code_law : ∀ pid (i : TileIndex [(parameters m words).block,8]),
      pid<(request m words).programs → i.1.val<(request m words).groups →
      Kernel.code op (parameters m words) (KernelGrid.atProgram (parameters m words) initial pid) i=codes pid i.1.val i.2.1.val)
    (scale_law : ∀ pid (g : Fin (parameters m words).block),
      pid<(request m words).programs → g.val<(request m words).groups →
      Kernel.scale op (parameters m words) (KernelGrid.atProgram (parameters m words) initial pid) g=scales pid g.val) :
    ∃ final, KernelGrid.run op (parameters m words) (List.range (request m words).programs) initial=some final ∧
      Result m words codes scales initial final := by
  let p := parameters m words
  let q := request m words
  let dest := fun pid => q.slot (pid/q.headCount)
  have metadata_read : ∀ pid,pid<q.programs → (initial.mem "D" (pid/p.heads)).read .int=(dest pid : Int) := by
    intro pid hp
    change (initial.mem "D" (pid/q.headCount)).read .int=(dest pid : Int)
    rw [metadata _ (program_token_bound q pid hp)]
    exact (Int.toNat_of_nonneg (h.bounds _ (program_token_bound q pid hp)).1).symm
  have unique : ∀ a b,a<q.programs → b<q.programs → KernelGridResult.target p dest a=KernelGridResult.target p dest b → a=b :=
    (checked_domain m words h).program_destination_unique
  exact KernelRaw.whole op p q.programs dest initial (groupBlock_covers q)
    metadata_read unique codes scales code_law scale_law

 theorem program_input {α : Type} [Inhabited α] (m : Inputs) (words : Nat → MetadataWord)
    (values : Nat → Nat → Nat → α) (initial : State α) (reads : Reads m words values initial)
    (pid g c : Nat) (hp : pid<(request m words).programs) (hg : g<(request m words).groups) (hc : c<8) :
    let s := KernelGrid.atProgram (parameters m words) initial pid
    (s.mem "K" (inputOffset (parameters m words) (s.pids 0*(parameters m words).heads+s.pids 1) g c)).read .real=values pid g c := by
  have he : pid/(parameters m words).heads*(parameters m words).heads+pid%(parameters m words).heads=pid := by
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div pid (parameters m words).heads
  simpa [KernelGrid.atProgram,he] using reads.input pid g c hp hg hc

 theorem native (a : NativeInstructions) (tree : ReductionTree) (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (values : Nat → Nat → Nat → Word) (initial : State Word)
    (reads : Reads m words values initial) :
    ∃ final, KernelGrid.run (nativeOps a tree) (parameters m words) (List.range (request m words).programs) initial=some final ∧
      Result m words (fun pid g c => (nativeCode a tree (values pid g) c).toInt)
        (fun pid g => nativeScale a tree (values pid g)) initial final := by
  apply whole (nativeOps a tree) m words h initial reads.destination
  · intro pid i hp hg
    exact (KernelModelResult.native a tree _ _ i.1 hg (values pid i.1.val)
      (fun c hc => program_input m words values initial reads pid i.1.val c hp hg hc)).2 i.2.1
  · intro pid g hp hg
    exact (KernelModelResult.native a tree _ _ g hg (values pid g.val)
      (fun c hc => program_input m words values initial reads pid g.val c hp hg hc)).1

 theorem mathematical (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (reads : Reads m words values initial) :
    ∃ final, KernelGrid.run KernelModels.realOps (parameters m words) (List.range (request m words).programs) initial=some final ∧
      Result m words (fun pid g c => (realCode (values pid g) c).toInt)
        (fun pid g => realScale (values pid g)) initial final := by
  apply whole KernelModels.realOps m words h initial reads.destination
  · intro pid i hp hg
    exact (KernelModelResult.mathematical _ _ i.1 hg (values pid i.1.val)
      (fun c hc => program_input m words values initial reads pid i.1.val c hp hg hc)).2 i.2.1
  · intro pid g hp hg
    exact (KernelModelResult.mathematical _ _ g hg (values pid g.val)
      (fun c hc => program_input m words values initial reads pid g.val c hp hg hc)).1

#axiomsClean whole
#axiomsClean native
#axiomsClean mathematical
#specNonCircular Reads avoiding [Kernel.kernel,FP.Structural.exec,KernelGrid.run]
#specNonCircular Result avoiding [Kernel.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelWhole
