import bench.tritonbench_g.lora_expand_gemv.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOBinding
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section

structure Bounds (g : Geometry) (bounds : RegionName → Nat) : Prop where
  index : ∀ b,b<g.B → b<bounds "I"
  x : ∀ b k,b<g.B → k<g.K → xoff g b k<bounds "X"
  w : ∀ l n k,l<g.L → n<g.N → k<g.K → woff g l n k<bounds "W"
  y : ∀ b n,b<g.B → n<g.N → yoff g b n<bounds "Y"

theorem bounds (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (order : List Nat) (vals : ChannelValues Word (inputs g)) (limits : RegionName → Nat)
    (h : (io rest plans g order).InBounds limits vals) : Bounds g limits := by
  constructor
  · intro b hb
    exact h.2.1 (0 : Fin 4) (⟨b,hb⟩,()) trivial
  · intro b k hb hk
    exact h.2.1 (1 : Fin 4) (⟨b,hb⟩,⟨k,hk⟩,()) trivial
  · intro l n k hl hn hk
    exact h.2.1 (2 : Fin 4) (⟨l,hl⟩,⟨n,hn⟩,⟨k,hk⟩,()) trivial
  · intro b n hb hn
    exact h.2.1 (3 : Fin 4) (⟨b,hb⟩,⟨n,hn⟩,()) trivial

theorem reads (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (order : List Nat) (s : State Word)
    (h : (io rest plans g order).InputCells (labeledInputs g raw) s) : IOGrid.Reads g raw s.mem := by
  constructor
  · intro b hb
    have hi := h (0 : Fin 4) (⟨b,hb⟩,()) trivial
    change s.mem "I" b=Cell.mk .int (logicalIndex g raw b) at hi
    rw [hi,Cell.read_same]
  · intro b k hb hk
    have hx := h (1 : Fin 4) (⟨b,hb⟩,⟨k,hk⟩,()) trivial
    simpa [io,memoryIO,inputs,labeledInputs,ProgramIO.InputCells,mapValue,entry] using hx
  · intro l n k hl hn hk
    have hw := h (2 : Fin 4) (⟨l,hl⟩,⟨n,hn⟩,⟨k,hk⟩,()) trivial
    simpa [io,memoryIO,inputs,labeledInputs,ProgramIO.InputCells,mapValue,entry] using hw

theorem y (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (order : List Nat) (s : State Word)
    (h : (io rest plans g order).InputCells (labeledInputs g raw) s)
    (b n : Nat) (hb : b<g.B) (hn : n<g.N) :
    s.mem "Y" (yoff g b n)=entry g raw "Y" (yoff g b n) := by
  have hy := h (3 : Fin 4) (⟨b,hb⟩,⟨n,hn⟩,()) trivial
  simpa [io,memoryIO,inputs,labeledInputs,ProgramIO.InputCells,mapValue,entry] using hy

theorem start (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (order : List Nat) (s : State Word)
    (h : (io rest plans g order).InputCells (labeledInputs g raw) s) :
    IOGrid.Snapshot plans g raw s.mem [] s.mem := by
  refine ⟨reads rest plans g raw order s h,?_,?_⟩
  · intro q hq
    have hc := coordinates g q hq
    simpa [offset] using y rest plans g raw order s h _ _ hc.2 hc.1
  · intro r o _; rfl

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOBinding
