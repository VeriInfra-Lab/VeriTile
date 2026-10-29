import bench.repairs.tritonbench_g.int8_matmul_quantization.FPEquiv

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Instances
noncomputable section
set_option maxHeartbeats 6000000
def scalarPlan (BK : Nat) : DotPlan BK where
  operations := (List.finRange BK).map Int8DequantMatmul.Micro.scalar
  fused := Bool.true
  complete := by
    have h (xs : List (Fin BK)) : ((xs.map Int8DequantMatmul.Micro.scalar).flatMap Int8DequantMatmul.Micro.terms)=xs := by
      induction xs with
      | nil => rfl
      | cons x xs ih => simp [Int8DequantMatmul.Micro.terms,ih]
    rw [h]
def selection (s : Shape) (c : Config) : Selection s c where
  plan := fun _ _ _ _ => scalarPlan c.BK
  origin := fun _ _ _ _ => ⟨0,0,0,1,0,2,by decide,by decide,by decide,by decide⟩
def invocation (s : Shape) (c : Config) : Invocation s :=
  ⟨c,selection s c,Schedule.ofPermutation s c (allCommits s c) (List.Perm.refl _)⟩
def reductionTree : List (Fin K) → Reduction K
  | [] => .zero
  | k::ks => .maximum (.lane k) (reductionTree ks)
theorem reductionTerms (xs : List (Fin K)) : (reductionTree xs).terms=xs := by
  induction xs with
  | nil => rfl
  | cons k ks ih => simp [reductionTree,Reduction.terms,ih]
def reductionPlan (K : Nat) : ReductionPlan K :=
  ⟨reductionTree (List.finRange K),by rw [reductionTerms]⟩
def quantInvocation (G K : Nat) : QuantInvocation G K :=
  ⟨fun _ => reductionPlan K,List.finRange G,List.Perm.refl _⟩
def shape : Shape := ⟨2,3,67⟩
def buffers (precision : Precision) : Buffers :=
  ⟨0,1,2,3,4,0,0,0,0,4,Bool.true,.half,.single,precision⟩
def quantBuffers : QuantBuffers := ⟨5,0,2,0,0,0,67,1,Bool.false,.half,Bool.true⟩
theorem separate (precision : Precision) : (buffers precision).Separate := by simp [Buffers.Separate,buffers]
theorem quantSeparate : quantBuffers.Separate := by simp [QuantBuffers.Separate,quantBuffers]
theorem compositionLink (precision : Precision) : CompositionLink quantBuffers (buffers precision) := by
  constructor <;> rfl
def bound (c : Config) (admitted : Source.Admitted c) (precision : Precision)
    (indices : IndexABI shape c (buffers precision)) : BoundCall ℝ ℝ ℝ shape (buffers precision) where
  invocation := invocation shape c
  parsedProduction := Source.productionAST
  sourceIdentity := rfl
  admitted := admitted
  indices := indices
  dispatch := List.Perm.refl _
  lowered := gridCode shape c (buffers precision) (selection shape c) (allCommits shape c)
  sourceLowering := rfl
def split1 : BoundCall ℝ ℝ ℝ shape (buffers .half) :=
  bound Source.config0 (by simp [Source.Admitted,Source.configs]) .half (by constructor <;> dsimp only [FormedRectangle.Safe] <;> decide)
def split2 : BoundCall ℝ ℝ ℝ shape (buffers .single) :=
  bound Source.config18 (by simp [Source.Admitted,Source.configs]) .single (by constructor <;> dsimp only [FormedRectangle.Safe] <;> decide)
def row : BoundQuantCall ℝ ℝ ℝ shape.M shape.K quantBuffers where
  invocation := quantInvocation shape.M shape.K
  parsedProduction := Source.productionAST
  sourceIdentity := rfl
  configIndex := 0
  block := 128
  indices := by constructor <;> dsimp only [FormedRectangle.Safe] <;> decide
  rowStructure := ⟨rfl,rfl,rfl⟩
  lowered := quantGridCode quantBuffers (fun _ => reductionPlan shape.K) (List.finRange shape.M)
  sourceLowering := rfl
def memoryABI (a : Ops I F H) : LocalEventABI I F H (Memory F H) a where
  observe := id
  step := fun e m => e.eval a m
  stepLaw := fun _ _ _ => rfl
def zeroABI (a : Ops I F H) : ZeroHookABI (memoryABI a) where
  zero := reset a
  law := fun _ _ _ => rfl
theorem realized_split_counts : split1.invocation.config.S=1 ∧ split2.invocation.config.S=2 := by decide
theorem realized_composition (precision : Precision) :
    quantBuffers.Separate ∧ (buffers precision).Separate ∧ CompositionLink quantBuffers (buffers precision) :=
  ⟨quantSeparate,separate precision,compositionLink precision⟩
theorem empty_wrapper (precision : Precision) (m : Memory ℝ ℝ) :
    (wrapper realOps ⟨0,3,0⟩ (buffers precision) 0 7 [] (invocation ⟨0,3,0⟩ Source.config18) m).currentDevice=7 := rfl
#axiomsClean scalarPlan
#axiomsClean reductionPlan
#axiomsClean split1
#axiomsClean split2
#axiomsClean row
#axiomsClean realized_composition
#axiomsClean empty_wrapper
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Instances
