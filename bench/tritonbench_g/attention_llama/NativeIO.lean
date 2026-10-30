import bench.tritonbench_g.attention_llama.Mathematical
import bench.tritonbench_g.attention_llama.Proofs.Lowering

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativeIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage NativeSchedule
noncomputable section
open scoped Classical

/-- Every physical CTA in the original two-dimensional launch grid. -/
abbrev Program (B : Nat) (c : Config) := Fin (count c.H) × Fin (B*c.NH)

def program (B : Nat) (c : Config) (i : Fin (Fintype.card (Program B c))) : Program B c :=
  (Fintype.equivFin (Program B c)).symm i

/-- Actual shared runtime-scalar source; the scalar ABI remains a register
argument, including the distinct constexpr-one argument case. -/
def invocation (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (p : Program B c) : Invocation (Term α) where
  kernel := Source.kernel c
  backend := Source.algebra
  arguments := [⟨"sm_scale",.real,[],fun _=>Native.argument C arg⟩]
  pids := fun ax=>if ax=0 then p.1.val else if ax=1 then p.2.val else 0

/-- Full ordinary grid. All programs observe the same caller input memory. -/
def schedule (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) : List (Invocation (Term α)) :=
  List.ofFn (fun i=>invocation C arg B c (program B c i))

/-- A CTA owns exactly its active query rows and output channels. This is
geometry only; source stores and pairwise disjointness are proved separately. -/
def writes (B : Nat) (c : Config) (p : Program B c) (r : RegionName) (o : Nat) : Prop :=
  ∃ i : TileIndex [B,c.NH,c.H,c.D],
    p.1.val=i.2.2.1.val/64 ∧ p.2.val=i.1.val*c.NH+i.2.1.val ∧
      r=c.Out ∧ o=IOLayout.output c i

/-- Finite FP16/E5M2 inputs and FP16 outputs, original strides, actual source,
raw scalar ABI, full ordinary grid, native lane observation and exact frame. -/
@[reducible] def io (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) : OrdinaryProgramIO α (Term α) where
  toMemoryIO := IOLayout.memory α B c
  schedule := fun _=>schedule C arg B c
  encode := Term.leaf
  observe := fun o=>Fin.cases (fun i=>Native.lower A c.fp8 c.D (lane i.2.2.2.1.val))
    (fun j=>Fin.elim0 j) o
  writes := fun _ k=>writes B c (program B c
    ⟨k.val,by simpa only [schedule,List.length_ofFn] using k.isLt⟩)

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativeIO
