import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Native
import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Layout
import bench.repairs.tritonbench_g.flash_attn.Contract

namespace VeriTile.Bench.Repairs.FlashAttn.KernelNative
open VeriTile Triton FP.Structural ActualSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000

/-- Original FP32 payloads are leaves, including signed zero and every NaN. -/
def initialMemory (memory : Memory Word) : Mem Expr :=
  liftMemory (fun r off _=>Node.word (memory r off))
def scalar (scale : Checked.ScalarABI) : Expr :=
  if scale.raw=.integer 1 then fun _=>.one else fun _=>.word scale.word

/-- Structural owner selection only. Output columns use their physical
reduction owner; L retains the admitted finite serialization winner. -/
def readOwner (x : Layout) (r : RegionName) (off : Nat) : Fin 4 :=
  if r="O" then
    if h : ∃ b h row col,(Work.output b h row col).active x.shape ∧
        off=FlashAttn.address x.o b h row col then
      Reduction.owner (Classical.choose (Classical.choose_spec (Classical.choose_spec (Classical.choose_spec h))))
    else 0
  else if r="L" then
    x.scratchOwner (off/x.shape.length/x.shape.heads) (off/x.shape.length%x.shape.heads) (off%x.shape.length)
  else 0

theorem output_owner (x : Layout) (injective : OutputInjective x) (b h row col : Nat)
    (active : (Work.output b h row col).active x.shape) :
    readOwner x "O" (FlashAttn.address x.o b h row col)=Reduction.owner col := by
  have hex : ∃ b' h' row' col',(Work.output b' h' row' col').active x.shape ∧
      FlashAttn.address x.o b h row col=FlashAttn.address x.o b' h' row' col' :=
    ⟨b,h,row,col,active,rfl⟩
  simp only [readOwner,ite_true,dif_pos hex]
  have chosen:=Classical.choose_spec (Classical.choose_spec (Classical.choose_spec (Classical.choose_spec hex)))
  exact congrArg Reduction.owner
    (injective _ _ _ _ b h row col chosen.1 active chosen.2.symm).2.2.2

#axiomsClean output_owner

def reference (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word) : Mem Expr :=
  referenceGrid algebra dot (parameters x) (scalar scale) (programs (parameters x)) (initialMemory memory)

def observe (I : Instructions) (x : Layout) (memory : Mem Expr) (r : RegionName) (off : Nat) : Word :=
  eval I ((memory r off).read .real (readOwner x r off))

def target (I : Instructions) (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word)
    (r : RegionName) (off : Nat) : Word := observe I x (reference x scale memory) r off

def run (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word) : Option (Mem Expr) :=
  runGrid algebra dot (parameters x) (scalar scale) (programs (parameters x)) (initialMemory memory)

theorem whole (I : Instructions) (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word) :
    ∃ final,run x scale memory=some final ∧ final=reference x scale memory ∧
      (∀ r off,observe I x final r off=target I x scale memory r off) ∧
      (∀ r off,¬Writes (parameters x) (programs (parameters x)) r off →
        final r off=initialMemory memory r off) := by
  refine ⟨reference x scale memory,source_grid _ _ _ _ _ _,rfl,fun _ _=>rfl,?_⟩
  exact fun r off h=>grid_frame _ _ _ _ _ _ r off h

theorem target_frame (I : Instructions) (x : Layout) (scale : Checked.ScalarABI) (memory : Memory Word)
    (r : RegionName) (off : Nat) (outside : ¬Writes (parameters x) (programs (parameters x)) r off) :
    target I x scale memory r off=memory (sourceRegion r) off := by
  unfold target observe reference
  rw [grid_frame _ _ _ _ _ _ r off outside]
  rfl

#specNonCircular target avoiding [FP.Structural.exec,FP.Structural.run,ActualSource.runGrid,run]
#axiomsClean whole
#axiomsClean target_frame
end
end VeriTile.Bench.Repairs.FlashAttn.KernelNative
