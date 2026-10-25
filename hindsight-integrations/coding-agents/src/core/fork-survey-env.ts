/**
 * Fork patch `survey-orca-env` (see the commit that added this file; bug-fix patch, no FORK-<NAME>.md).
 *
 * The codebase survey is a detached background job and must not pass itself off as the agent in the
 * parent's terminal pane. Orca (a terminal / agent manager) identifies a pane's agent by its ORCA_*
 * variables. A survey `claude -p` that inherits them reports "this pane's agent finished" through
 * Orca's own Stop hook when it exits, and Orca then closes the parent's terminal, killing whatever
 * unattended run was going on there. HINDSIGHT_DISABLE_HOOKS only silences Hindsight's own hooks, so
 * the survey's environment drops every variable whose name starts with ORCA_ and keeps the rest.
 */
export function withoutTerminalManagerEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(base).filter(([k]) => !k.startsWith("ORCA_")));
}
