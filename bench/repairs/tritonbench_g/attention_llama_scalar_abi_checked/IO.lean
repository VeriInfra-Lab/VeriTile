import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealSourceTransport
import VeriTile.Triton.Memory.ParameterizedAlgorithmProgramIO
import bench.repairs.tritonbench_g.attention_llama.Proofs.HostResult
import bench.repairs.tritonbench_g.attention_llama.Proofs.Geometry
import bench.tritonbench_g.attention_llama.Proofs.FP8RealMemory
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.IOLayout

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealLayout
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
open VeriTile.Bench.TritonBenchG.AttentionLlama
noncomputable section

abbrev QueryIndex (B : Nat) (c : Config) := TileIndex [B,c.NH,c.H,c.D]
abbrev KeyIndex (B : Nat) (c : Config) := TileIndex [B,c.NH,c.N,c.D]

def queryIndex (B : Nat) (c : Config) : Fin (Fintype.card (QueryIndex B c))→QueryIndex B c :=
  (Fintype.equivFin (QueryIndex B c)).symm

def keyIndex (B : Nat) (c : Config) : Fin (Fintype.card (KeyIndex B c))→KeyIndex B c :=
  (Fintype.equivFin (KeyIndex B c)).symm

def querySlot (B : Nat) (c : Config) : QueryIndex B c→Fin (Fintype.card (QueryIndex B c)) :=
  Fintype.equivFin (QueryIndex B c)

def keySlot (B : Nat) (c : Config) : KeyIndex B c→Fin (Fintype.card (KeyIndex B c)) :=
  Fintype.equivFin (KeyIndex B c)

@[simp] theorem queryIndex_slot (B : Nat) (c : Config) (i : QueryIndex B c) :
    queryIndex B c (querySlot B c i)=i := Equiv.symm_apply_apply _ _
@[simp] theorem keyIndex_slot (B : Nat) (c : Config) (i : KeyIndex B c) :
    keyIndex B c (keySlot B c i)=i := Equiv.symm_apply_apply _ _

/-- Mathematical interpretation of the exact source's FP16/byte input roles.
Only the selected K/V representation is active; the four caller strides and
full finite extents remain visible. No private memory is introduced. -/
@[reducible] def memory (B : Nat) (c : Config) : AlgorithmMemoryIO where
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [c.Q,c.K,c.V,c.Out]
  ity := fun i=>if i.val<3 then .float else .int
  iarity := Fin.cases (Fintype.card (QueryIndex B c)) (fun _=>Fintype.card (KeyIndex B c))
  ibuf := Fin.cases c.Q (Fin.cases c.K (Fin.cases c.V (Fin.cases c.K (Fin.cases c.V Fin.elim0))))
  oty := fun _=>.float
  oarity := fun _=>Fintype.card (QueryIndex B c)
  obuf := fun _=>c.Out
  obuf_mem := by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := Fin.cases (fun _ _ _ _ j=>IOLayout.query c (queryIndex B c j))
    (Fin.cases (fun _ _ _ _ j=>IOLayout.key c (keyIndex B c j))
      (Fin.cases (fun _ _ _ _ j=>IOLayout.value c (keyIndex B c j))
        (Fin.cases (fun _ _ _ _ j=>IOLayout.key c (keyIndex B c j))
          (Fin.cases (fun _ _ _ _ j=>IOLayout.value c (keyIndex B c j)) (fun i=>Fin.elim0 i)))))
  imask := fun i _ _ _ _ _=>
    ((i.val=1 ∨ i.val=2) → c.fp8=Bool.false) ∧
      ((i.val=3 ∨ i.val=4) → c.fp8=Bool.true)
  owin := fun _ _ _ _ _ j=>IOLayout.output c (queryIndex B c j)
  omask := fun _ _ _ _ _ _=>True
  swin := fun i=>Fin.elim0 i
  smask := fun i=>Fin.elim0 i

/-- A key byte is interpreted by Real only when some actual physical query
program reaches its masked block. Exterior cells and untouched tail blocks
have no finiteness condition. -/
def reachedKey (c : Config) (k : Fin c.N) : Prop :=
  ∃ p : Fin (count c.H), k.val/64*64<MaskCoverage.stop c.N c.causal c.SP p.val

def domain (B : Nat) (c : Config) (vals : (memory B c).Ctx) : Prop :=
  c.fp8=Bool.true → ∀ i : KeyIndex B c, reachedKey c i.2.2.1 →
    FP8RealMemory.FiniteByte (vals 3 (keySlot B c i)) ∧
      FP8RealMemory.FiniteByte (vals 4 (keySlot B c i))

/-- Exact E5M2 decoding as a mathematical real number; domain excludes only
reached nonfinite encodings, as required by the existing Real theorem. -/
def decode (byte : Int) : ℝ :=
  ((Float8E5Bits.ofInt byte).decodeRat.getD 0 : ℚ)

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealLayout

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealMathematical
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid HostResult MaskCoverage
noncomputable section

/-- Independent finite Q/K/V arrays; byte inputs use exact E5M2 decoding in
Real and the other source branch retains the original FP16 interpretation. -/
def inputs (B : Nat) (c : Config) (vals : (RealLayout.memory B c).Ctx) :
    HostResult.Inputs B c.NH c.H c.D where
  query := fun b h q d=>vals 0 (RealLayout.querySlot B c (b,h,q,d,()))
  key := fun b h k d=>if hk:k<c.N then
    if c.fp8 then RealLayout.decode (vals 3 (RealLayout.keySlot B c (b,h,⟨k,hk⟩,d,())))
    else vals 1 (RealLayout.keySlot B c (b,h,⟨k,hk⟩,d,())) else 0
  value := fun b h k d=>if hk:k<c.N then
    if c.fp8 then RealLayout.decode (vals 4 (RealLayout.keySlot B c (b,h,⟨k,hk⟩,d,())))
    else vals 2 (RealLayout.keySlot B c (b,h,⟨k,hk⟩,d,())) else 0

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealMathematical

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealIO
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
noncomputable section

/-- The authoritative runtime-scalar source, interpreted mathematically. -/
def source (c : Config) : AlgorithmSource :=
  ⟨(Source.kernel c).eraseDType,RealSourceTransport.projection_supported c⟩

/-- Every physical query block and packed batch/head program is launched. -/
def grid (B : Nat) (c : Config) : Grid := ⟨[count c.H,B*c.NH,1]⟩

/-- Finite strided Q/K/V and output windows, the actual source and full
ordinary grid. The runtime scalar is bound to its original register; neither
the source nor the caller's memory is specialized or replaced. -/
@[reducible] def io (B : Nat) (c : Config) (scale : ℝ) : ParameterizedAlgorithmProgramIO :=
  {RealLayout.memory B c with
    initialUndef:=.zero
    stages:=fun _=>[.ordinary (source c) (grid B c)]
    arguments:=fun _=>[⟨"sm_scale",.real,[],Tile.scalar (some scale : WithBot ℝ)⟩]}

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealIO

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.LogicalIO
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama HostResult MaskCoverage
noncomputable section

/-- Logical mathematical arrays plus the complete heterogeneous source windows.
The connection keeps inactive byte/float roles and all alias constraints. -/
structure Inputs (B : Nat) (c : Config) where
  arrays : HostResult.Inputs B c.NH c.H c.D
  backing : (RealLayout.memory B c).Ctx
  connection : arrays = RealMathematical.inputs B c backing

def Inputs.query (xs : Inputs B c) := xs.arrays.query
def Inputs.key (xs : Inputs B c) := xs.arrays.key
def Inputs.value (xs : Inputs B c) := xs.arrays.value

abbrev Outputs (B : Nat) (c : Config) := Fin B → Fin c.NH → Fin c.H → Fin c.D → ℝ

def inputsEquiv (B : Nat) (c : Config) : (RealLayout.memory B c).Ctx ≃ Inputs B c where
  toFun := fun raw => ⟨RealMathematical.inputs B c raw,raw,rfl⟩
  invFun := Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,backing,connection⟩
    cases connection
    rfl

def outputsEquiv (B : Nat) (c : Config) :
    ((o : Fin (RealLayout.memory B c).nOut) →
      Fin ((RealLayout.memory B c).oarity o) → ((RealLayout.memory B c).oty o).carrier) ≃ Outputs B c where
  toFun := fun values b h row col => values 0 (RealLayout.querySlot B c (b,h,row,col,()))
  invFun := fun values _ j =>
    let q := RealLayout.queryIndex B c j
    values q.1 q.2.1 q.2.2.1 q.2.2.2.1
  left_inv := by
    intro values
    funext o j
    have zero : o = 0 := Fin.eq_zero o
    subst o
    have eta (q : RealLayout.QueryIndex B c) :
        (q.1,q.2.1,q.2.2.1,q.2.2.2.1,()) = q := by
      rcases q with ⟨b,h,row,col,⟨⟩⟩
      rfl
    simp only [eta,RealLayout.querySlot,RealLayout.queryIndex,Equiv.apply_symm_apply]
  right_inv := by
    intro values
    funext b h row col
    simp only [RealLayout.queryIndex_slot]

def domain (B : Nat) (c : Config) (xs : Inputs B c) : Prop :=
  RealLayout.domain B c xs.backing

def io (B : Nat) (c : Config) (scale : ℝ) :
    LogicalDomainIO (RealIO.io B c scale).Ctx
      ((o : Fin (RealIO.io B c scale).nOut) →
        Fin ((RealIO.io B c scale).oarity o) → ((RealIO.io B c scale).oty o).carrier)
      (Inputs B c) (Outputs B c) (RealIO.io B c scale).ImplementsOn where
  inputs := inputsEquiv B c
  outputs := outputsEquiv B c

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.LogicalIO
