export default async function workflow(host) {
  // Reusable Cloudflare security-audit workflow (skill: ~/.agents/skills/security-audit).
  // Phases: recon -> parallel hunting -> independent verification -> synthesis.
  // Studio is out of scope unless args.includeStudio is true.
  const target = host.args?.target ?? process.cwd?.() ?? ".";
  const skill = host.args?.skillDir ?? "~/.agents/skills/security-audit";
  const scope = host.args?.includeStudio ? "including studio/" : "excluding studio/ (deferred per owner)";
  const evidenceSchema = { type: "object", required: ["complete", "evidence", "unresolved"], properties: { complete: { type: "boolean" }, evidence: { type: "array", items: { type: "string" } }, unresolved: { type: "array", items: { type: "string" } } } };
  const hunters = await host.parallel([
    { label: "recon", schema: evidenceSchema, input: `Read ${skill}/SKILL.md and RECONNAISSANCE.md. Phase 1 recon of repo at ${target}, ${scope}: architecture, trust boundaries, input surfaces. Inspect source bodies. Return complete/evidence/unresolved.` },
    { label: "hunt-network-config", schema: evidenceSchema, input: `Hunter (ATTACK-CLASSES, PROTOCOLS-RPC-AND-MESSAGING, WEB-PROTOCOL-AND-AUTH) for ${target}, ${scope}: network bridges, RON config, deserialization, unbounded allocation. Inspect bodies. Report candidates with file:line, principal, boundary, concrete result, or no-findings with checks.` },
    { label: "hunt-core-dos", schema: evidenceSchema, input: `Hunter (MEMORY-SAFETY-AND-BINARY, RESOURCE-EXHAUSTION-AND-AVAILABILITY) for ${target}: unsafe, panic-on-input, overflow, unbounded loops/budgets in libs/tasks. Inspect bodies.` },
    { label: "hunt-supply-ci", schema: evidenceSchema, input: `Hunter (SUPPLY-CHAIN-AND-RELEASE, CLOUD-AND-DEPLOYMENT, CLIENT-SIDE) for ${target}, ${scope}: workflows, manifests, lockfiles, CI permissions. Inspect files.` },
  ]);
  const verifier = await host.agent({ label: "independent-verifier", schema: evidenceSchema, input: `Independent verifier: try to disprove each candidate in ${hunters.map(h => h?.ref ?? "null").join(", ")} by re-inspecting source at ${target}. Classify confirmed (severity per skill anchors) / needs_validation (no severity) / rejected. Build coverage ledger: frontend, APIs, auth, databases, infrastructure, dependencies, core-libs, tasks — mark covered/partial/N/A with evidence.` });
  const synthesis = await host.agent({ label: "synthesis", schema: evidenceSchema, input: `Synthesize verified findings from ${verifier?.ref ?? "null"}: posture before fixes, priorities, coverage gaps. Do not invent severity for needs_validation.` });
  const report = { status: "complete", target, scope, hunters: hunters.map(h => h?.ref ?? null), verifier: verifier?.ref ?? null, synthesis: synthesis?.text ?? null, notes: [hunters, verifier, synthesis].flatMap(r => r?.notes ? [r.notes] : []) };
  return { summary: JSON.stringify(report) };
}
