import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphComplete

/-! Candidate reduction trees recovered with a last-writer trace policy in the isolated
Triton 3.7.1 SM90 captures. These are source graphs before explicit FMA
contraction, not reductions of already rounded products claimed native-equal.
The companion bounded PTX trace report records the source, artifact hashes,
profiles and tested metadata. BK8/BK16 have competing shared writers;
those trees are candidate choices, not established unique native graphs. No
GPU numerical validation is claimed. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Equational

def sm90Plan1 : ReductionPlan 1 where
  tree := .input ⟨0, by decide⟩
  padding := 0
  valid := by decide +kernel

def sm90Plan2 : ReductionPlan 2 where
  tree := .add (.input ⟨0, by decide⟩) (.input ⟨1, by decide⟩)
  padding := 0
  valid := by decide +kernel

def sm90Plan4 : ReductionPlan 4 where
  tree := .add (.add (.input ⟨0, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨3, by decide⟩))
  padding := 0
  valid := by decide +kernel

def sm90Plan8 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨7, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨1, by decide⟩))) (.add (.add (.input ⟨6, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨0, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Plan16 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩))) (.add (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)))) (.add (.add (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩))) (.add (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Plan64 : ReductionPlan 64 where
  tree := .add (.add (.add (.add (.add (.add (.input ⟨0, by decide⟩) (.input ⟨16, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨24, by decide⟩))) (.add (.add (.input ⟨4, by decide⟩) (.input ⟨20, by decide⟩)) (.add (.input ⟨12, by decide⟩) (.input ⟨28, by decide⟩)))) (.add (.add (.add (.input ⟨2, by decide⟩) (.input ⟨18, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨26, by decide⟩))) (.add (.add (.input ⟨6, by decide⟩) (.input ⟨22, by decide⟩)) (.add (.input ⟨14, by decide⟩) (.input ⟨30, by decide⟩))))) (.add (.add (.add (.add (.input ⟨1, by decide⟩) (.input ⟨17, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨25, by decide⟩))) (.add (.add (.input ⟨5, by decide⟩) (.input ⟨21, by decide⟩)) (.add (.input ⟨13, by decide⟩) (.input ⟨29, by decide⟩)))) (.add (.add (.add (.input ⟨3, by decide⟩) (.input ⟨19, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨27, by decide⟩))) (.add (.add (.input ⟨7, by decide⟩) (.input ⟨23, by decide⟩)) (.add (.input ⟨15, by decide⟩) (.input ⟨31, by decide⟩)))))) (.add (.add (.add (.add (.add (.input ⟨32, by decide⟩) (.input ⟨48, by decide⟩)) (.add (.input ⟨40, by decide⟩) (.input ⟨56, by decide⟩))) (.add (.add (.input ⟨36, by decide⟩) (.input ⟨52, by decide⟩)) (.add (.input ⟨44, by decide⟩) (.input ⟨60, by decide⟩)))) (.add (.add (.add (.input ⟨34, by decide⟩) (.input ⟨50, by decide⟩)) (.add (.input ⟨42, by decide⟩) (.input ⟨58, by decide⟩))) (.add (.add (.input ⟨38, by decide⟩) (.input ⟨54, by decide⟩)) (.add (.input ⟨46, by decide⟩) (.input ⟨62, by decide⟩))))) (.add (.add (.add (.add (.input ⟨33, by decide⟩) (.input ⟨49, by decide⟩)) (.add (.input ⟨41, by decide⟩) (.input ⟨57, by decide⟩))) (.add (.add (.input ⟨37, by decide⟩) (.input ⟨53, by decide⟩)) (.add (.input ⟨45, by decide⟩) (.input ⟨61, by decide⟩)))) (.add (.add (.add (.input ⟨35, by decide⟩) (.input ⟨51, by decide⟩)) (.add (.input ⟨43, by decide⟩) (.input ⟨59, by decide⟩))) (.add (.add (.input ⟨39, by decide⟩) (.input ⟨55, by decide⟩)) (.add (.input ⟨47, by decide⟩) (.input ⟨63, by decide⟩))))))
  padding := 0
  valid := by decide +kernel

/-- Only captured block lengths select a candidate tree. The fallback is
an explicit software schedule; no native correspondence is claimed for it. -/
def sm90Plans (n : Nat) : ReductionPlan n :=
  if h : n = 1 then h.symm ▸ sm90Plan1 else
  if h : n = 2 then h.symm ▸ sm90Plan2 else
  if h : n = 4 then h.symm ▸ sm90Plan4 else
  if h : n = 8 then h.symm ▸ sm90Plan8 else
  if h : n = 16 then h.symm ▸ sm90Plan16 else
  if h : n = 64 then h.symm ▸ sm90Plan64 else
  ReductionPlan.withSeed n

#axiomsClean sm90Plan1
#axiomsClean sm90Plan2
#axiomsClean sm90Plan4
#axiomsClean sm90Plan8
#axiomsClean sm90Plan16
#axiomsClean sm90Plan64
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
