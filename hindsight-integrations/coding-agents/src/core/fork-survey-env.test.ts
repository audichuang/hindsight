// Fork patch `survey-orca-env`: the detached survey must not impersonate the parent's Orca pane.
// Fork-owned test file (FORK.md: no edits to upstream test files). The startCodebaseSurvey cases fail
// if the hook in survey.ts is lost in a rebase; the unit cases pin the filter itself.
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import { withoutTerminalManagerEnv } from "./fork-survey-env";
import { resolveHostConfig } from "./host-client";
import { startCodebaseSurvey } from "./survey";
import { SURVEY_SPEC_ENV } from "./survey-lease";

vi.mock("./host-client", () => ({ resolveHostConfig: vi.fn() }));

describe("withoutTerminalManagerEnv", () => {
  it("drops every ORCA_ variable and keeps everything else, values untouched", () => {
    const out = withoutTerminalManagerEnv({
      ORCA_PANE_KEY: "pane-1",
      ORCA_AGENT_HOOK_TOKEN: "secret",
      ORCA_: "bare-prefix",
      PATH: "/usr/bin",
      ORCASTRATE: "no-underscore",
      MY_ORCA_VAR: "not-a-prefix",
      orca_lower: "case-sensitive",
      EMPTY: "",
    });
    expect(out).toEqual({
      PATH: "/usr/bin",
      ORCASTRATE: "no-underscore",
      MY_ORCA_VAR: "not-a-prefix",
      orca_lower: "case-sensitive",
      EMPTY: "",
    });
  });

  it("does not mutate its input", () => {
    const base = { ORCA_PANE_KEY: "pane-1", PATH: "/usr/bin" };
    withoutTerminalManagerEnv(base);
    expect(base).toEqual({ ORCA_PANE_KEY: "pane-1", PATH: "/usr/bin" });
  });
});

describe("startCodebaseSurvey: terminal-manager env isolation", () => {
  const saved = { ...process.env };
  let lockDir: string;

  beforeEach(() => {
    lockDir = mkdtempSync(join(tmpdir(), "hindsight-fork-survey-env-"));
    vi.mocked(resolveHostConfig).mockReturnValue({
      cfg: resolveConfig({ apiUrl: "https://api.example.test", apiToken: "test-token" }),
      bankId: "bank-1",
    });
  });
  afterEach(() => {
    for (const k of Object.keys(process.env)) if (!(k in saved)) delete process.env[k];
    Object.assign(process.env, saved);
    vi.restoreAllMocks();
    rmSync(lockDir, { recursive: true, force: true });
  });

  function fakeSpawn() {
    return vi.fn().mockImplementation(() => {
      const child = Object.assign(new EventEmitter(), {
        pid: process.pid,
        unref: vi.fn(),
        kill: vi.fn(),
      });
      queueMicrotask(() => child.emit("spawn"));
      return child;
    });
  }

  for (const harness of ["claude-code", "codex", "antigravity-cli", "opencode"] as const) {
    it(`${harness}: the survey child gets no ORCA_* var and keeps the rest`, async () => {
      process.env.ORCA_PANE_KEY = "pane-1";
      process.env.ORCA_AGENT_HOOK_PORT = "1234";
      process.env.ORCA_AGENT_HOOK_TOKEN = "secret";
      process.env.PATH = "/usr/bin:/bin";
      process.env.HINDSIGHT_API_URL = "http://example.invalid:1";
      process.env.ORCASTRATE = "no-underscore-kept";
      const spawn = fakeSpawn();
      await startCodebaseSurvey("/repo", {
        harness,
        lease: { dir: lockDir },
        supervisorPath: "/x/survey-supervisor.js",
        mcpServerPath: "/x/mcp-server.js",
        claudeBin: "/bin/claude",
        spawn,
        exists: () => true,
        opencodeDefinesAgent: () => true,
      });
      expect(spawn).toHaveBeenCalledTimes(1);
      const env = spawn.mock.calls[0][2].env as NodeJS.ProcessEnv;
      expect(env[SURVEY_SPEC_ENV]).toBeTruthy(); // really the survey launch, not some other spawn
      expect(Object.keys(env).filter((k) => k.startsWith("ORCA_"))).toEqual([]);
      expect(env.PATH).toBe("/usr/bin:/bin");
      expect(env.HINDSIGHT_API_URL).toBe("http://example.invalid:1");
      expect(env.HINDSIGHT_DISABLE_HOOKS).toBe("1");
      expect(env.ORCASTRATE).toBe("no-underscore-kept");
    });
  }
});
