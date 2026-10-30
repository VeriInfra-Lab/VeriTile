import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

export type ScopeTask = {
  task: string; proof_scope: 'selected' | 'pending'; body_status: string;
  entries: string[]; python: string; kernel: string;
};
export type ReleaseScope = {
  root_count: number; source_dependency_count: number; tasks: ScopeTask[];
  proof_task_counts: { selected: number; pending: number };
  pending_original_program_reviews: number;
  known_unselected_build_failures: string[];
  pending_python_tests: Record<string, string>;
  body_counts: { tasks: number; examples: number; task_status: Record<string, number> };
  examples: { name: string; python: string; lean: string; status: string; source_provenance: string }[];
};

// Fail on source/scope drift. This reports the selected release inventory;
// building a website does not run Lean or certify the pending original programs.
export const releaseScope: ReleaseScope = JSON.parse(
  execFileSync('python3', ['bench/verified_scope.py', '--json'], {
    cwd: resolve(process.cwd(), '..'), encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
  }),
);
export const sourceLink = (path: string) =>
  `https://github.com/VeriInfra-Lab/VeriTile/blob/main/${path}`;
