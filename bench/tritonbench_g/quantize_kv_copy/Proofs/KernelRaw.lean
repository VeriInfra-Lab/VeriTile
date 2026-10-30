import bench.tritonbench_g.quantize_kv_copy.Proofs.KernelGridResult
import bench.tritonbench_g.quantize_kv_copy.Proofs.KernelModelResult

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelRaw
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000

def Result {α : Type} (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (initial final : State α) : Prop :=
  (∀ pid g c,pid<n → g<p.groups → c<8 →
    final.mem "O" (codeOffset p (KernelGridResult.target p dest pid) g c)=.mk .int (codes pid g c)) ∧
  (∀ pid g,pid<n → g<p.groups →
    final.mem "S" (scaleOffset p (KernelGridResult.target p dest pid) g)=.mk .real (scales pid g)) ∧
  (∀ (r : RegionName) o,
    (r≠"O" ∨ ∀ pid g c,pid<n → g<p.groups → c<8 →
      o≠codeOffset p (KernelGridResult.target p dest pid) g c) →
    (r≠"S" ∨ ∀ pid g,pid<n → g<p.groups →
      o≠scaleOffset p (KernelGridResult.target p dest pid) g) →
    final.mem r o=initial.mem r o)

theorem whole {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) (n : Nat)
    (dest : Nat → Nat) (initial : State α) (covers : p.groups≤p.block)
    (metadata_read : ∀ pid,pid<n → (initial.mem "D" (pid/p.heads)).read .int=(dest pid : Int))
    (unique : ∀ a b,a<n → b<n → KernelGridResult.target p dest a=KernelGridResult.target p dest b → a=b)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (code_law : ∀ pid (i : TileIndex [p.block,8]),pid<n → i.1.val<p.groups →
      Kernel.code op p (KernelGrid.atProgram p initial pid) i=codes pid i.1.val i.2.1.val)
    (scale_law : ∀ pid (g : Fin p.block),pid<n → g.val<p.groups →
      Kernel.scale op p (KernelGrid.atProgram p initial pid) g=scales pid g.val) :
    ∃ final,KernelGrid.run op p (List.range n) initial=some final ∧
      Result p n dest codes scales initial final := by
  let order := List.range n
  refine ⟨KernelGridResult.finalState op p order initial,KernelGridResult.run_finalState op p order initial,?_,?_,?_⟩
  · intro pid g c hp hg hc
    let i : TileIndex [p.block,8] := (⟨g,lt_of_lt_of_le hg covers⟩,⟨c,hc⟩,PUnit.unit)
    have result := KernelGridResult.final_code op p initial n dest metadata_read unique order
      (fun _ hp => List.mem_range.mp hp) List.nodup_range pid (List.mem_range.mpr hp) i hg
    rw [code_law pid i hp hg] at result
    exact result
  · intro pid g hp hg
    let i : TileIndex [p.block] := (⟨g,lt_of_lt_of_le hg covers⟩,PUnit.unit)
    have result := KernelGridResult.final_scale op p initial n dest metadata_read unique order
      (fun _ hp => List.mem_range.mp hp) List.nodup_range pid (List.mem_range.mpr hp) i hg
    rw [scale_law pid i.1 hp hg] at result
    exact result
  · intro r o codes_miss scales_miss
    exact KernelGridResult.final_frame op p initial n dest metadata_read order
      (fun _ hp => List.mem_range.mp hp) r o codes_miss scales_miss initial ⟨fun _ => rfl,fun _ => rfl⟩

/-- Raw source input reads and signed destination indices only. -/
structure Reads {α : Type} [Inhabited α] (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (values : Nat → Nat → Nat → α) (initial : State α) : Prop where
  input : ∀ pid g c,pid<n → g<p.groups → c<8 →
    (initial.mem "K" (inputOffset p pid g c)).read .real=values pid g c
  destination : ∀ pid,pid<n → (initial.mem "D" (pid/p.heads)).read .int=(dest pid : Int)

theorem program_input {α : Type} [Inhabited α] (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (values : Nat → Nat → Nat → α) (initial : State α) (reads : Reads p n dest values initial)
    (pid g c : Nat) (hp : pid<n) (hg : g<p.groups) (hc : c<8) :
    let s := KernelGrid.atProgram p initial pid
    (s.mem "K" (inputOffset p (s.pids 0*p.heads+s.pids 1) g c)).read .real=values pid g c := by
  have he : pid/p.heads*p.heads+pid%p.heads=pid := by
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div pid p.heads
  simpa [KernelGrid.atProgram,he] using reads.input pid g c hp hg hc

 theorem native (a : NativeInstructions) (tree : ReductionTree) (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (covers : p.groups≤p.block)
    (unique : ∀ a b,a<n → b<n → KernelGridResult.target p dest a=KernelGridResult.target p dest b → a=b) (values : Nat → Nat → Nat → Word) (initial : State Word)
    (reads : Reads p n dest values initial) :
    ∃ final, KernelGrid.run (nativeOps a tree) p (List.range n) initial=some final ∧
      Result p n dest (fun pid g c => (nativeCode a tree (values pid g) c).toInt)
        (fun pid g => nativeScale a tree (values pid g)) initial final := by
  apply whole (nativeOps a tree) p n dest initial covers reads.destination unique
  · intro pid i hp hg
    exact (KernelModelResult.native a tree _ _ i.1 hg (values pid i.1.val)
      (fun c hc => program_input p n dest values initial reads pid i.1.val c hp hg hc)).2 i.2.1
  · intro pid g hp hg
    exact (KernelModelResult.native a tree _ _ g hg (values pid g.val)
      (fun c hc => program_input p n dest values initial reads pid g.val c hp hg hc)).1

 theorem mathematical (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (covers : p.groups≤p.block)
    (unique : ∀ a b,a<n → b<n → KernelGridResult.target p dest a=KernelGridResult.target p dest b → a=b) (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (reads : Reads p n dest values initial) :
    ∃ final, KernelGrid.run KernelModels.realOps p (List.range n) initial=some final ∧
      Result p n dest (fun pid g c => (realCode (values pid g) c).toInt)
        (fun pid g => realScale (values pid g)) initial final := by
  apply whole KernelModels.realOps p n dest initial covers reads.destination unique
  · intro pid i hp hg
    exact (KernelModelResult.mathematical _ _ i.1 hg (values pid i.1.val)
      (fun c hc => program_input p n dest values initial reads pid i.1.val c hp hg hc)).2 i.2.1
  · intro pid g hp hg
    exact (KernelModelResult.mathematical _ _ g hg (values pid g.val)
      (fun c hc => program_input p n dest values initial reads pid g.val c hp hg hc)).1


#axiomsClean mathematical
#axiomsClean native
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelRaw
