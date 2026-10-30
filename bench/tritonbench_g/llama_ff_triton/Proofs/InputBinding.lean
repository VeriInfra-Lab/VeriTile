import bench.tritonbench_g.llama_ff_triton.IO

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Reads (xs : Inputs α) (s : State α) : Prop where
  input : ∀ o, (s.mem "A" o).read .fp16 = xs (0 : Fin 7) o
  w1Half : ∀ o, (s.mem "W1" o).read .fp16 = xs (1 : Fin 7) o
  w3Half : ∀ o, (s.mem "W3" o).read .fp16 = xs (2 : Fin 7) o
  rmsHalf : ∀ o, (s.mem "RMS" o).read .fp16 = xs (3 : Fin 7) o
  w1Byte : ∀ o, (s.mem "W1" o).read .int = xs (4 : Fin 7) o
  w3Byte : ∀ o, (s.mem "W3" o).read .int = xs (5 : Fin 7) o
  rmsByte : ∀ o, (s.mem "RMS" o).read .int = xs (6 : Fin 7) o

theorem input_reads (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (s : State α)
    (hin : (programIO native a g p).InputBinding xs s) : Reads xs s :=
  ⟨hin (0 : Fin 7),hin (1 : Fin 7),hin (2 : Fin 7),hin (3 : Fin 7),hin (4 : Fin 7),hin (5 : Fin 7),hin (6 : Fin 7)⟩

theorem Reads.bytes {xs : Inputs α} {s : State α} (h : Reads xs s) (hd : byteDomain xs) :
    ByteValues s := by
  intro r hr o
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl | rfl | rfl
  · rw [h.rmsByte]; exact hd.2.2 o
  · rw [h.w1Byte]; exact hd.1 o
  · rw [h.w3Byte]; exact hd.2.1 o

theorem Reads.real_target {xs : Inputs ℝ} {s : State ℝ} (h : Reads xs s)
    (p : Parameters) (row column : Nat) :
    Real.target p s row column =
      LlamaFF.realTarget ⟨p.inner,p.epsilon⟩ (Reference.realInput p xs row) column := by
  have hr : Real.readWeight p s "RMS" = Reference.decodedWeight p (xs (3 : Fin 7)) (xs (6 : Fin 7)) := by
    funext o
    simp only [Real.readWeight,Reference.decodedWeight,h.rmsByte,h.rmsHalf]
  dsimp only [Real.target,Real.parameters,Real.inputs,Real.readWeight,Reference.realInput,
    Reference.decodedWeight]
  rw [hr]
  simp only [h.input,h.w1Half,h.w3Half,h.rmsHalf,h.w1Byte,h.w3Byte,h.rmsByte]



theorem output_frame {β : Type} [Inhabited β] (native : Bool) (a : Arithmetic β)
    (g : GateInstructions β) (p : Parameters) (xs : Inputs β) (initial final : State β)
    (hf : ∀ r o, (r≠("OUT" : RegionName) ∨ ∀ x : Grid.Index p, True → o≠Grid.address p x) →
      final.mem r o=initial.mem r o) : (programIO native a g p).Frame xs initial final := by
  intro r o ho
  apply hf r o
  by_cases hr : r=("OUT" : RegionName)
  · subst r
    right
    intro x _
    exact (ho.1 (0 : Fin 1) (x.1,x.2,()) trivial).resolve_left (by simp [programIO,outputPorts])
  · exact Or.inl hr

#axiomsClean Reads.bytes
#axiomsClean Reads.real_target
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
