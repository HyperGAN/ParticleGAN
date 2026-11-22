/* ParticleGAN Forge Leaderboard — Static Space App
 * Loads live data & training GIFs directly from Hugging Face Dataset:
 * https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence
 */

(() => {
  "use strict";

  const DEFAULT_DATASET_REPO = "HyperGAN/particlegan-forge-evidence";
  const GITHUB_REPO = "HyperGAN/ParticleGAN";
  const DEFAULT_BRANCH = "develop";

  const params =
    typeof window !== "undefined" && window.location
      ? new URLSearchParams(window.location.search)
      : new URLSearchParams("");
  const CONFIG = {
    datasetRepo: params.get("dataset") || DEFAULT_DATASET_REPO,
    requestedRev: params.get("rev") || "main",
    mediaRev: params.get("media_rev") || null,
    dataPath: params.get("data_path") || "data/leaderboard.json",
  };

  const state = {
    raw: null,
    meta: {
      datasetRepo: CONFIG.datasetRepo,
      hfRevision: CONFIG.requestedRev,
      mediaRevision: CONFIG.mediaRev || CONFIG.requestedRev,
      gitCommit: "",
      generatedAt: "",
      view: "discriminator_stability",
      viewRevision: null,
      declaredViewRevisions: {},
      fullSuiteList: "",
      fullSuiteDated: "",
      fullSuiteSha256: "",
      inputDigest: "",
      backfillAttempted: null,
      backfillTotal: null,
      receiptRowsTotal: null,
      countedViewReqs: null,
      distinctRequiredTests: null,
      fullSuiteTotal: 69,
      stretchAssignments: null,
      distinctStretchTests: null,
      heldTests: null,
      heldAssignments: null,
      removedTests: null,
      removedAssignments: null,
      energyBenchmarkRows: 0,
      receiptsWithEnergy: 0,
      caveats: [],
    },
    families: [],
    receipts: [],
    energyBenchmarks: [],
    taskDefinitions: new Map(),
    selectedFamilyId: null,
    selectedReceiptKey: null,
    selectedEnergyKey: null,
    lbMode: "all",
    lbSort: { key: "total_pass", dir: "desc" },
    energySort: { key: "grid100_j_to_pass", dir: "asc" },
    energyFilter: "all",
    energySearch: "",
    expSort: { key: "tier", dir: "asc" },
    expFilters: {
      family: "",
      tier: "",
      status: "",
      role: "",
      search: "",
      hasGif: false,
    },
    expLimit: 100,
    gifLayout: "compare",
    gifTask: "",
    gifLeft: "",
    gifRight: "",
    gifOnlyAvailable: false,
  };

  /* ---------- Utilities ---------- */

  const $ = (id) => document.getElementById(id);

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function shortHash(value, len = 12) {
    if (!value) return "—";
    const s = String(value).trim();
    return s.length > len ? s.slice(0, len) : s;
  }

  function hfResolveUrl(path, revOverride = null) {
    if (!path) return null;
    if (/^https?:\/\//i.test(path)) return path;
    const cleaned = String(path).replace(/^\/+/, "");
    const rev = revOverride || state.meta.mediaRevision || state.meta.hfRevision || "main";
    return `https://huggingface.co/datasets/${state.meta.datasetRepo}/resolve/${encodeURIComponent(rev)}/${cleaned}`;
  }

  function githubBlobUrl(relPath, refOverride = null) {
    const ref = refOverride || state.meta.gitCommit || DEFAULT_BRANCH;
    const cleaned = String(relPath || "").replace(/^\/+/, "");
    return `https://github.com/${GITHUB_REPO}/blob/${encodeURIComponent(ref)}/${cleaned}`;
  }

  function toInt(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function toNullableNum(value) {
    if (value == null || value === "" || typeof value === "boolean") return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatNum(value, digits = 2, suffix = "") {
    if (value == null || !Number.isFinite(value)) return "—";
    const formatted = value >= 1000
      ? value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })
      : value.toFixed(digits);
    return suffix ? `${formatted}${suffix}` : formatted;
  }

  function formatJoules(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    if (value >= 1000) {
      return `${value.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })} J`;
    }
    return `${value.toFixed(1)} J`;
  }

  function formatStateBpp(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    if (value === 0) return "0";
    if (value < 0.001) return value.toExponential(1);
    return value.toFixed(value < 0.1 ? 4 : 2);
  }

  function normalizeStatus(rawStatus) {
    const s = String(rawStatus ?? "not_run").trim();
    const lower = s.toLowerCase();
    if (lower === "pass" || lower === "passed") return "PASS";
    if (lower === "fail" || lower === "failed") return "FAIL";
    if (lower === "blocked" || lower === "refused" || lower === "unsupported") return "blocked";
    if (lower === "incompatible") return "incompatible";
    if (lower === "error" || lower === "timeout" || lower === "invalid" || lower === "incomplete") return "error";
    if (lower === "not_run" || lower === "not run" || lower === "not_measured" || lower === "not measured" || lower === "unknown" || lower === "unmeasured" || lower === "") {
      return "not_run";
    }
    return lower;
  }

  function normalizeRole(rawRole) {
    const r = String(rawRole ?? "required").trim().toLowerCase();
    if (r.includes("stretch")) return "stretch";
    if (r.includes("held")) return "held";
    if (r.includes("remov")) return "removed";
    if (r.includes("diag")) return "diagnostic";
    return "required";
  }

  function normalizeGifPath(rawPath) {
    if (!rawPath) return null;
    const s = String(rawPath).trim();
    if (!s) return null;
    if (/^https?:\/\//i.test(s)) return s;
    if (s.startsWith("/")) {
      // Absolute local workstation path cannot be fetched from HF directly
      return null;
    }
    if (s.startsWith("media/")) return s;
    if (s.startsWith("reports/forge/")) return `media/forge/${s.slice("reports/forge/".length)}`;
    if (s.startsWith("reports/")) return `media/${s.slice("reports/".length)}`;
    if (s.startsWith("../")) {
      const stripped = s.replace(/^(\.\.\/)+/, "");
      return stripped.startsWith("media/") ? stripped : `media/forge/${stripped}`;
    }
    if (s.endsWith(".gif") || s.endsWith(".png") || s.endsWith(".webp")) {
      return `media/forge/${s}`;
    }
    return null;
  }

  function statusBadgeHtml(status) {
    const norm = normalizeStatus(status);
    const css = `badge-${norm.toLowerCase()}`;
    return `<span class="badge ${escapeHtml(css)}">${escapeHtml(norm)}</span>`;
  }

  function roleBadgeHtml(role) {
    const norm = normalizeRole(role);
    return `<span class="badge badge-${escapeHtml(norm)}">${escapeHtml(norm)}</span>`;
  }

  function ratioCellHtml(pass, total, notes = []) {
    const pct = total > 0 ? Math.max(0, Math.min(100, Math.round((pass / total) * 100))) : 0;
    const sub = notes.filter(Boolean).join(", ");
    return `
      <div class="ratio-main">${pass}/${total}</div>
      ${sub ? `<span class="ratio-sub">${escapeHtml(sub)}</span>` : ""}
      ${total > 0 ? `<div class="bar-track" title="${pct}%"><div class="bar-fill" style="width:${pct}%"></div></div>` : ""}
    `;
  }

  /* ---------- Schema Normalization ---------- */

  function extractTierStats(fam, tierNum) {
    const key = String(tierNum);
    const tierObj = (fam.tiers && (fam.tiers[key] || fam.tiers[`tier_${key}`] || fam.tiers[`tier${key}`])) || {};
    const counts = tierObj.counts || {};
    const pass = toInt(
      tierObj.passed ?? tierObj.pass ?? counts.PASS ?? fam[`tier${key}_pass`] ?? fam[`tier_${key}_pass`],
      0
    );
    const total = toInt(
      tierObj.total ?? fam[`tier${key}_total`] ?? fam[`tier_${key}_total`],
      0
    );
    const notRun = toInt(tierObj.not_run ?? counts.UNKNOWN ?? fam[`tier${key}_not_run`], 0);
    const blocked = toInt(tierObj.blocked ?? counts.BLOCKED ?? fam[`tier${key}_blocked`], 0);
    const incompatible = toInt(tierObj.incompatible ?? counts.INCOMPATIBLE ?? fam[`tier${key}_incompatible`], 0);
    const diagnostic = toInt(tierObj.diagnostic ?? tierObj.diagnostic_attempts ?? fam[`tier${key}_diagnostic`], 0);
    const notes = [];
    if (notRun > 0) notes.push(`${notRun} not run`);
    if (blocked > 0) notes.push(`${blocked} blocked`);
    if (incompatible > 0) notes.push(`${incompatible} incompat.`);
    if (diagnostic > 0) notes.push(`${diagnostic} diag.`);
    return { pass, total, notRun, blocked, incompatible, diagnostic, notes };
  }

  function normalizeFamily(rawFam, idx) {
    const id = String(rawFam.id || rawFam.family_id || rawFam.family || `family-${idx}`);
    const label = String(rawFam.label || rawFam.name || rawFam.family_label || id);
    const t1 = extractTierStats(rawFam, 1);
    const t2 = extractTierStats(rawFam, 2);
    const t3 = extractTierStats(rawFam, 3);
    const t4 = extractTierStats(rawFam, 4);

    const totalObj =
      rawFam.total_view_requirements || rawFam.total || rawFam.totals || rawFam.view_requirements || {};
    const totalPass = toInt(
      totalObj.passed ?? totalObj.pass ?? rawFam.total_pass ?? rawFam.view_pass,
      t1.pass + t2.pass + t3.pass + t4.pass
    );
    const totalReq = toInt(
      totalObj.total ?? rawFam.total_required ?? rawFam.view_total,
      t1.total + t2.total + t3.total + t4.total
    );
    const totalNotes = [];
    const totalNotRun = toInt(totalObj.not_run ?? rawFam.total_not_run, t1.notRun + t2.notRun + t3.notRun + t4.notRun);
    const totalBlocked = toInt(totalObj.blocked ?? rawFam.total_blocked, t1.blocked + t2.blocked + t3.blocked + t4.blocked);
    const totalIncompat = toInt(
      totalObj.incompatible ?? rawFam.total_incompatible,
      t1.incompatible + t2.incompatible + t3.incompatible + t4.incompatible
    );
    const totalDiag = toInt(totalObj.diagnostic ?? totalObj.diagnostic_attempts ?? rawFam.total_diagnostic, t1.diagnostic + t2.diagnostic + t3.diagnostic + t4.diagnostic);
    if (totalNotRun > 0) totalNotes.push(`${totalNotRun} not run`);
    if (totalBlocked > 0) totalNotes.push(`${totalBlocked} blocked`);
    if (totalIncompat > 0) totalNotes.push(`${totalIncompat} incompat.`);
    if (totalDiag > 0) totalNotes.push(`${totalDiag} diag.`);

    const fs = rawFam.full_suite || rawFam.fullSuite || {};
    const fullPass = toInt(fs.pass ?? fs.passed ?? fs.PASS ?? rawFam.full_suite_pass, 0);
    const fullFail = toInt(fs.fail ?? fs.failed ?? fs.FAIL ?? rawFam.full_suite_fail, 0);
    const fullNotMeasured = toInt(fs.not_measured ?? fs.not_run ?? fs.unmeasured ?? rawFam.full_suite_not_measured, 0);
    const fullOther = toInt(fs.other ?? fs.blocked ?? fs.other_outcomes ?? rawFam.full_suite_other, 0);
    const fullTotal = toInt(fs.total ?? rawFam.full_suite_total, fullPass + fullFail + fullNotMeasured + fullOther);
    const fullHeld = fs.held || null;
    const fullOtherBreakdown = fs.other_breakdown || null;

    const st = rawFam.stretch || rawFam.stretch_counts || {};
    const stTotalObj = st.total && typeof st.total === "object" ? st.total : st;
    const stretchPass = toInt(stTotalObj.passed ?? stTotalObj.pass ?? st.total_pass ?? rawFam.stretch_pass, 0);
    const stretchTotal = toInt(
      typeof st.total === "number" ? st.total : stTotalObj.total ?? rawFam.stretch_total,
      0
    );
    const stretchNotes = [];
    if (toInt(stTotalObj.not_run, 0) > 0) stretchNotes.push(`${toInt(stTotalObj.not_run, 0)} not run`);
    if (toInt(stTotalObj.incompatible, 0) > 0) stretchNotes.push(`${toInt(stTotalObj.incompatible, 0)} incompat.`);
    if (toInt(stTotalObj.diagnostic, 0) > 0) stretchNotes.push(`${toInt(stTotalObj.diagnostic, 0)} diag.`);

    const pagePath = rawFam.page || rawFam.family_page || `reports/forge/families/${id}.md`;

    return {
      id,
      label,
      selectedRecipe: rawFam.selected_recipe || rawFam.recipe_label || rawFam.candidate_id || "",
      candidateId: rawFam.candidate_id || rawFam.selected_candidate_id || "",
      candidateRevision: rawFam.candidate_revision || "",
      recipeHash: rawFam.recipe_sha256 || rawFam.recipe_hash || "",
      sourceHash: rawFam.source_digest || rawFam.source_hash || rawFam.source_sha256 || rawFam.cohort_source_hash || "",
      sourceOriginCommit: rawFam.source_origin_commit || "",
      cohortAnchor: rawFam.cohort_anchor || rawFam.current_cohort_anchor || rawFam.cohort || "",
      backend: rawFam.backend || "",
      stale: Boolean(rawFam.stale || rawFam.is_stale),
      staleNote: rawFam.stale_note || "",
      qualifiedTier: rawFam.qualified_tier ?? null,
      qualificationNote: rawFam.qualification_note || rawFam.qualification || "",
      holdoutClaim: rawFam.holdout_claim || "",
      tuningBudget: rawFam.tuning_budget || rawFam.tuning || fs.tuning_budget || "",
      pagePath,
      t1,
      t2,
      t3,
      t4,
      tier1_pass: t1.pass,
      tier2_pass: t2.pass,
      tier3_pass: t3.pass,
      tier4_pass: t4.pass,
      total_pass: totalPass,
      total_req: totalReq,
      totalNotes,
      full_pass: fullPass,
      full_fail: fullFail,
      full_not_measured: fullNotMeasured,
      full_other: fullOther,
      full_total: fullTotal,
      fullHeld,
      fullOtherBreakdown,
      stretch_pass: stretchPass,
      stretch_total: stretchTotal,
      stretchNotes,
      r256_joules_per_step: toNullableNum(rawFam.r256_joules_per_step),
      ring16_j_to_pass: toNullableNum(rawFam.ring16_j_to_pass),
      grid100_j_to_pass: toNullableNum(rawFam.grid100_j_to_pass),
      tuning: rawFam.tuning_budget || rawFam.qualification_note || "",
      raw: rawFam,
    };
  }

  function normalizeEnergyObj(rawEnergy) {
    if (!rawEnergy || typeof rawEnergy !== "object") return null;
    const joulesTotal = toNullableNum(rawEnergy.joules_total);
    const joulesPerStep = toNullableNum(rawEnergy.joules_per_step);
    const avgPowerWatts = toNullableNum(rawEnergy.avg_power_watts);
    const firstPassStep = toNullableNum(rawEnergy.first_pass_step);
    const sustainedPassStep = toNullableNum(rawEnergy.sustained_pass_step);
    const joulesToFirstPass = toNullableNum(rawEnergy.joules_to_first_pass);
    const joulesToSustainedPass = toNullableNum(rawEnergy.joules_to_sustained_pass);
    const energyBackend = rawEnergy.energy_backend ? String(rawEnergy.energy_backend) : null;
    const gpuName = rawEnergy.gpu_name ? String(rawEnergy.gpu_name) : null;
    const contended = typeof rawEnergy.contended === "boolean" ? rawEnergy.contended : null;
    if (
      joulesTotal == null &&
      joulesPerStep == null &&
      avgPowerWatts == null &&
      joulesToFirstPass == null &&
      joulesToSustainedPass == null
    ) {
      return null;
    }
    return {
      joules_total: joulesTotal,
      joules_per_step: joulesPerStep,
      avg_power_watts: avgPowerWatts,
      first_pass_step: firstPassStep,
      sustained_pass_step: sustainedPassStep,
      joules_to_first_pass: joulesToFirstPass,
      joules_to_sustained_pass: joulesToSustainedPass,
      energy_backend: energyBackend,
      gpu_name: gpuName,
      contended,
    };
  }

  function normalizeEnergyBenchmark(rawRow, idx) {
    const candidateId = String(rawRow.candidate_id || rawRow.id || `candidate-${idx}`);
    const sourceFile = String(rawRow.source_file || "");
    const sourceCommit = rawRow.source_commit ? String(rawRow.source_commit) : "";
    const hardwareGpu = rawRow.hardware_gpu ? String(rawRow.hardware_gpu) : "";
    const evidenceScope = String(rawRow.evidence_scope || "study_energy_benchmark");
    const contended = typeof rawRow.contended === "boolean" ? rawRow.contended : null;
    return {
      key: `${candidateId}::${hardwareGpu}::${sourceCommit}::${sourceFile}::${idx}`,
      candidate_id: candidateId,
      family_id: rawRow.family_id ? String(rawRow.family_id) : "",
      optimizer_family: rawRow.optimizer_family ? String(rawRow.optimizer_family) : "",
      hardware_gpu: hardwareGpu,
      contended,
      source_commit: sourceCommit,
      source_file: sourceFile,
      r256_ms_per_step: toNullableNum(rawRow.r256_ms_per_step),
      r256_watts: toNullableNum(rawRow.r256_watts),
      r256_joules_per_step: toNullableNum(rawRow.r256_joules_per_step),
      state_bytes_per_param: toNullableNum(rawRow.state_bytes_per_param),
      ring16_first_pass_step: toNullableNum(rawRow.ring16_first_pass_step),
      ring16_j_to_pass: toNullableNum(rawRow.ring16_j_to_pass),
      grid100_first_pass_step: toNullableNum(rawRow.grid100_first_pass_step),
      grid100_sustained_pass_step: toNullableNum(rawRow.grid100_sustained_pass_step),
      grid100_j_to_pass: toNullableNum(rawRow.grid100_j_to_pass),
      grid100_j_to_sustained_pass: toNullableNum(rawRow.grid100_j_to_sustained_pass),
      staggered100_first_pass_step: toNullableNum(rawRow.staggered100_first_pass_step),
      staggered100_j_to_pass: toNullableNum(rawRow.staggered100_j_to_pass),
      occam_pass: toNullableNum(rawRow.occam_pass),
      occam_executed: toNullableNum(rawRow.occam_executed),
      evidence_scope: evidenceScope,
      raw: rawRow,
    };
  }

  function normalizeReceipt(rawItem, familyLookup, idx) {
    const familyId = String(rawItem.family_id || rawItem.family || rawItem.trainer_family || "unknown");
    const fam = familyLookup.get(familyId);
    const familyLabel = String(rawItem.family_label || (fam && fam.label) || familyId);
    const taskId = String(rawItem.task_id || rawItem.task || rawItem.experiment || rawItem.id || `task-${idx}`);
    const tier = rawItem.tier != null && rawItem.tier !== "" ? toInt(rawItem.tier, 0) : 0;
    const role = normalizeRole(rawItem.role || rawItem.disposition || (rawItem.is_stretch ? "stretch" : rawItem.is_held ? "held" : "required"));
    const status = normalizeStatus(rawItem.status || rawItem.outcome || rawItem.verdict || rawItem.result);
    const gifPath = normalizeGifPath(
      rawItem.gif_url || rawItem.gif_path || rawItem.gif_relpath || rawItem.media_path || rawItem.media_url || rawItem.training_gif
    );

    return {
      key: `${familyId}::${taskId}::${idx}`,
      family_id: familyId,
      family_label: familyLabel,
      task_id: taskId,
      tier,
      role,
      views: Array.isArray(rawItem.views) ? rawItem.views : [],
      status,
      recorded_status: String(rawItem.recorded_status || ""),
      reason: String(rawItem.reason || rawItem.coverage_reason || rawItem.blocker || rawItem.summary || ""),
      candidate_id: String(rawItem.candidate_id || (fam && fam.candidateId) || ""),
      recipe_hash: String(rawItem.recipe_hash || rawItem.recipe_sha256 || (fam && fam.recipeHash) || ""),
      source_hash: String(rawItem.source_hash || rawItem.source_sha256 || (fam && fam.sourceHash) || ""),
      task_sha256: String(rawItem.task_sha256 || ""),
      attempt_id: String(rawItem.attempt_id || ""),
      wall_seconds: typeof rawItem.wall_seconds === "number" ? rawItem.wall_seconds : null,
      energy: normalizeEnergyObj(rawItem.energy || (rawItem.cost && rawItem.cost.energy) || rawItem),
      cohort: String(rawItem.cohort || (fam && fam.cohortAnchor) || ""),
      gif_path: gifPath,
      metrics: rawItem.metrics || rawItem.metrics_summary || null,
      raw: rawItem,
    };
  }

  function normalizeDataset(raw, hfApiSha) {
    const meta = state.meta;
    const rawMeta = raw.metadata || raw.provenance || raw;

    meta.datasetRepo = rawMeta.dataset_repo || CONFIG.datasetRepo;
    meta.hfRevision =
      CONFIG.requestedRev !== "main"
        ? CONFIG.requestedRev
        : raw.hf_revision || rawMeta.hf_revision || hfApiSha || "main";
    meta.mediaRevision =
      CONFIG.mediaRev || raw.media_hf_revision || rawMeta.media_hf_revision || meta.hfRevision;
    meta.gitCommit = raw.git_commit || rawMeta.git_commit || rawMeta.source_commit || "";
    meta.generatedAt = raw.generated_at || rawMeta.generated_at || raw.published_at || "";
    meta.view = raw.view || rawMeta.view || "discriminator_stability";
    meta.viewRevision = raw.view_revision ?? rawMeta.view_revision ?? null;
    meta.declaredViewRevisions = raw.declared_view_revisions || rawMeta.declared_view_revisions || {};

    const rawFsList = raw.full_suite_list || rawMeta.full_suite_list || null;
    if (rawFsList && typeof rawFsList === "object") {
      meta.fullSuiteList = rawFsList.id || "technique-inventory-full-suite-2026-11-18";
      meta.fullSuiteDated = rawFsList.dated || "";
      meta.fullSuiteSha256 = rawFsList.sha256 || "";
    } else {
      meta.fullSuiteList = rawFsList || "technique-inventory-full-suite-2026-11-18";
      meta.fullSuiteDated = "";
      meta.fullSuiteSha256 = raw.full_suite_sha256 || rawMeta.full_suite_sha256 || "";
    }
    meta.inputDigest = raw.input_digest || rawMeta.input_digest || "";

    const summary = raw.summary || raw.kpis || raw.overview || {};
    const attemptCov = summary.attempt_coverage || {};
    meta.countedViewReqs = summary.counted_view_requirements ?? summary.view_requirements ?? null;
    meta.distinctRequiredTests = summary.distinct_required_tests ?? null;
    meta.fullSuiteTotal =
      summary.full_suite_tests ??
      summary.full_suite_total ??
      (rawFsList && typeof rawFsList === "object" ? rawFsList.count : null) ??
      69;
    meta.stretchAssignments = summary.stretch_assignments ?? summary.stretch_view_assignments ?? null;
    meta.distinctStretchTests = summary.stretch_distinct_tests ?? summary.distinct_stretch_tests ?? summary.stretch_tests ?? null;
    meta.heldTests = summary.held_distinct_tests ?? summary.held_tests ?? null;
    meta.heldAssignments = summary.held_assignments ?? null;
    meta.removedTests = summary.removed_distinct_tests ?? summary.removed_tests ?? null;
    meta.removedAssignments = summary.removed_assignments ?? null;
    meta.receiptRowsTotal = summary.receipt_rows ?? null;
    meta.energyBenchmarkRows = summary.energy_benchmark_rows ?? null;
    meta.receiptsWithEnergy = summary.receipts_with_energy ?? 0;
    meta.backfillAttempted = attemptCov.attempted ?? summary.backfill_attempted ?? null;
    meta.backfillTotal = attemptCov.total ?? summary.backfill_total ?? null;
    meta.caveats = Array.isArray(raw.caveats) ? raw.caveats : Array.isArray(summary.caveats) ? summary.caveats : [];

    // Families list
    const rawFamilies =
      raw.families || raw.selected_recipes || raw.leaderboard || raw.selected_families || [];
    const families = rawFamilies.map((f, i) => normalizeFamily(f, i));
    const familyLookup = new Map();
    for (const f of families) {
      familyLookup.set(f.id, f);
      familyLookup.set(f.label, f);
    }

    // Task definitions (optional metadata map)
    const taskDefinitions = new Map();
    const rawDefs = raw.task_definitions || raw.held_and_removed || raw.problems || [];
    if (Array.isArray(rawDefs)) {
      for (const def of rawDefs) {
        const tid = def.task_id || def.id || def.test;
        if (tid) taskDefinitions.set(String(tid), def);
      }
    } else if (rawDefs && typeof rawDefs === "object") {
      for (const [tid, def] of Object.entries(rawDefs)) {
        taskDefinitions.set(String(tid), def);
      }
    }

    // Receipts / family x task matrix
    const rawReceipts = raw.receipts || raw.tasks || raw.task_results || raw.results || [];
    let receipts = [];
    if (Array.isArray(rawReceipts)) {
      receipts = rawReceipts.map((item, idx) => normalizeReceipt(item, familyLookup, idx));
    }

    // Fallback: if receipts were nested under each family's `tasks`
    if (receipts.length === 0) {
      let idx = 0;
      for (const f of families) {
        const nested = f.raw.tasks || f.raw.receipts;
        if (Array.isArray(nested)) {
          for (const item of nested) {
            receipts.push(normalizeReceipt({ family_id: f.id, family_label: f.label, ...item }, familyLookup, idx++));
          }
        } else if (nested && typeof nested === "object") {
          for (const [taskId, item] of Object.entries(nested)) {
            receipts.push(normalizeReceipt({ family_id: f.id, family_label: f.label, task_id: taskId, ...item }, familyLookup, idx++));
          }
        }
      }
    }

    // Optional separate media map: { "family_id::task_id": "media/forge/..." } or { family_id: { task_id: "media/..." } }
    const mediaMap = raw.media || raw.gifs || raw.gif_index || null;
    if (mediaMap && typeof mediaMap === "object") {
      for (const r of receipts) {
        if (!r.gif_path) {
          const direct = mediaMap[`${r.family_id}::${r.task_id}`] || mediaMap[`${r.family_id}/${r.task_id}`];
          const nested = mediaMap[r.family_id] && mediaMap[r.family_id][r.task_id];
          const found = normalizeGifPath(direct || nested);
          if (found) r.gif_path = found;
        }
      }
    }

    // Infer KPI numbers if not explicitly given
    if (meta.countedViewReqs == null && families.length > 0) {
      meta.countedViewReqs = Math.max(...families.map((f) => f.total_req), 0) || null;
    }
    if (meta.stretchAssignments == null && families.length > 0) {
      meta.stretchAssignments = Math.max(...families.map((f) => f.stretch_total), 0) || null;
    }

    // Energy benchmarks (optional array in leaderboard.json)
    const rawEnergy = Array.isArray(raw.energy_benchmarks) ? raw.energy_benchmarks : [];
    const energyBenchmarks = rawEnergy.map((row, idx) => normalizeEnergyBenchmark(row, idx));
    if (meta.energyBenchmarkRows == null) {
      meta.energyBenchmarkRows = energyBenchmarks.length;
    }

    state.raw = raw;
    state.families = families;
    state.receipts = receipts;
    state.energyBenchmarks = energyBenchmarks;
    state.taskDefinitions = taskDefinitions;
  }

  /* ---------- Rendering ---------- */

  function setStatus(message, kind = "loading") {
    const banner = $("status-banner");
    const msg = $("status-message");
    if (!message) {
      banner.classList.add("hidden");
      return;
    }
    banner.classList.remove("hidden", "error");
    const spinner = banner.querySelector(".spinner");
    if (kind === "error") {
      banner.classList.add("error");
      if (spinner) spinner.classList.add("hidden");
    } else {
      if (spinner) spinner.classList.remove("hidden");
    }
    msg.textContent = message;
  }

  function renderProvenanceBar() {
    const m = state.meta;
    const dsUrl = `https://huggingface.co/datasets/${m.datasetRepo}`;
    $("prov-dataset").textContent = m.datasetRepo;
    $("prov-dataset").href = dsUrl;
    $("link-hf-dataset").href = dsUrl;
    $("prov-hf-rev").textContent = shortHash(m.hfRevision, 12);
    $("prov-hf-rev").title = m.hfRevision || "";
    $("prov-git-commit").textContent = m.gitCommit ? shortHash(m.gitCommit, 10) : "develop";
    $("prov-git-commit").title = m.gitCommit || "";
    $("prov-view").textContent = m.viewRevision != null ? `${m.view} rev ${m.viewRevision}` : m.view;
    $("prov-updated").textContent = m.generatedAt ? m.generatedAt.replace("T", " ").replace("Z", " UTC") : "—";

    const dataUrl = hfResolveUrl(CONFIG.dataPath, m.hfRevision);
    $("footer-data-url").href = dataUrl;
    $("footer-data-url").textContent = `${CONFIG.dataPath} @ ${shortHash(m.hfRevision, 10)} ↗`;

    if (m.gitCommit) {
      $("link-inventory").href = githubBlobUrl("reports/forge/technique-inventory.md", m.gitCommit);
      $("link-problems").href = githubBlobUrl("reports/forge/problems.md", m.gitCommit);
    }
  }

  function renderKpis() {
    const m = state.meta;
    const famCount = state.families.length;
    const gifCount = state.receipts.filter((r) => Boolean(r.gif_path)).length;
    const distinctTasks = new Set(state.receipts.map((r) => r.task_id)).size;

    const heldDistinct = m.heldTests ?? new Set(state.receipts.filter((r) => r.role === "held").map((r) => r.task_id)).size;
    const removedDistinct = m.removedTests ?? new Set(state.receipts.filter((r) => r.role === "removed").map((r) => r.task_id)).size;
    const stretchDistinct = m.distinctStretchTests ?? new Set(state.receipts.filter((r) => r.role === "stretch").map((r) => r.task_id)).size;

    $("kpi-families").textContent = String(famCount);
    $("kpi-families-sub").textContent = `${famCount} selected recipes · 1 per family`;

    $("kpi-view-reqs").textContent = m.countedViewReqs != null ? String(m.countedViewReqs) : "—";
    $("kpi-view-reqs-sub").textContent = m.distinctRequiredTests != null
      ? `Across ${m.distinctRequiredTests} distinct required tests`
      : "Counted view requirements (shared tests counted per view)";

    $("kpi-full-suite").textContent = String(m.fullSuiteTotal ?? 69);
    $("kpi-full-suite-sub").textContent = m.fullSuiteList
      ? `${m.fullSuiteList}${m.fullSuiteSha256 ? ` · ${shortHash(m.fullSuiteSha256, 8)}` : ""}`
      : "Fixed test list · failures included";

    $("kpi-stretch").textContent = m.stretchAssignments != null ? String(m.stretchAssignments) : String(stretchDistinct || "—");
    $("kpi-stretch-sub").textContent = stretchDistinct
      ? `${stretchDistinct} distinct stretch tests · same protocol`
      : "Extra-credit & frontier tasks";

    $("kpi-held").textContent = `${heldDistinct} held · ${removedDistinct} rem.`;
    $("kpi-held-sub").textContent =
      m.heldAssignments != null && m.removedAssignments != null
        ? `${m.heldAssignments} held + ${m.removedAssignments} removed view assignments · visible in full-suite`
        : "Excluded from tier counts; failures remain in full-suite";

    $("kpi-receipts").textContent = `${state.receipts.length.toLocaleString()} / ${gifCount}`;
    $("kpi-receipts-sub").textContent = m.backfillAttempted != null && m.backfillTotal != null
      ? `${distinctTasks} tasks · backfill ${m.backfillAttempted}/${m.backfillTotal} attempts`
      : `${distinctTasks} tasks · ${gifCount} with published GIFs`;

    const uncontendedEnergy = state.energyBenchmarks.filter((row) => row.contended !== true);
    const pool = uncontendedEnergy.length > 0 ? uncontendedEnergy : state.energyBenchmarks;
    const withGrid = pool.filter((row) => row.grid100_j_to_pass != null);
    const withRing = pool.filter((row) => row.ring16_j_to_pass != null);
    const withR256 = pool.filter((row) => row.r256_joules_per_step != null);
    if (withGrid.length > 0) {
      const bestGrid = [...withGrid].sort((a, b) => a.grid100_j_to_pass - b.grid100_j_to_pass)[0];
      const bestR256 = withR256.length > 0
        ? [...withR256].sort((a, b) => a.r256_joules_per_step - b.r256_joules_per_step)[0]
        : null;
      $("kpi-energy").textContent = formatJoules(bestGrid.grid100_j_to_pass);
      $("kpi-energy-sub").textContent = `${bestGrid.candidate_id} (grid100)` +
        (bestR256 ? ` · min r256 ${formatNum(bestR256.r256_joules_per_step, 2)} J/step` : "") +
        ` · ${state.energyBenchmarks.length} cohort rows`;
    } else if (withRing.length > 0) {
      const bestRing = [...withRing].sort((a, b) => a.ring16_j_to_pass - b.ring16_j_to_pass)[0];
      $("kpi-energy").textContent = formatJoules(bestRing.ring16_j_to_pass);
      $("kpi-energy-sub").textContent = `${bestRing.candidate_id} (ring16) · ${state.energyBenchmarks.length} cohort rows`;
    } else if (withR256.length > 0) {
      const bestR256 = [...withR256].sort((a, b) => a.r256_joules_per_step - b.r256_joules_per_step)[0];
      $("kpi-energy").textContent = `${formatNum(bestR256.r256_joules_per_step, 2)} J/step`;
      $("kpi-energy-sub").textContent = `${bestR256.candidate_id} (r256) · ${state.energyBenchmarks.length} cohort rows`;
    } else {
      $("kpi-energy").textContent = "—";
      $("kpi-energy-sub").textContent = "Hardware NVML · no energy rows loaded";
    }

    if (m.caveats && m.caveats.length > 0) {
      $("caveat-list").innerHTML = m.caveats.map((c) => `<li>${escapeHtml(c)}</li>`).join("");
    }
  }

  function renderLeaderboard() {
    const table = $("leaderboard-table");
    const tbody = $("leaderboard-tbody");
    table.dataset.lbMode = state.lbMode;

    // Sort indicator on headers
    table.querySelectorAll("thead th.sortable").forEach((th) => {
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.sort === state.lbSort.key) {
        th.classList.add(state.lbSort.dir === "asc" ? "sort-asc" : "sort-desc");
      }
    });

    if (state.families.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12" class="empty-cell">No family records available in dataset.</td></tr>`;
      return;
    }

    const sorted = [...state.families].sort((a, b) => {
      const k = state.lbSort.key;
      const dir = state.lbSort.dir === "asc" ? 1 : -1;
      const va = a[k];
      const vb = b[k];
      if (typeof va === "number" && typeof vb === "number") {
        if (va !== vb) return (va - vb) * dir;
        return a.label.localeCompare(b.label);
      }
      return String(va ?? "").localeCompare(String(vb ?? "")) * dir;
    });

    tbody.innerHTML = sorted
      .map((f) => {
        const selectedCls = state.selectedFamilyId === f.id ? "selected-row" : "";
        const staleBadge = f.stale ? `<span class="badge badge-stale" title="Recorded source measurement is stale; no current qualification">stale</span>` : "";
        const recipeLabel = f.selectedRecipe && f.selectedRecipe !== f.label ? `<code>${escapeHtml(f.selectedRecipe)}</code>` : "";
        const sourcePill = f.sourceHash ? `<code title="Source hash: ${escapeHtml(f.sourceHash)}">src:${escapeHtml(shortHash(f.sourceHash, 8))}</code>` : "";
        const tuningText = [f.qualificationNote, f.tuningBudget].filter(Boolean).join(" · ") || "—";

        return `
          <tr class="clickable ${selectedCls}" data-family-id="${escapeHtml(f.id)}">
            <td class="family-cell">
              <div class="family-name">${escapeHtml(f.label)}</div>
              <div class="family-meta">${recipeLabel} ${sourcePill} ${staleBadge}</div>
            </td>
            <td data-col-group="tiers" class="num">${ratioCellHtml(f.t1.pass, f.t1.total, f.t1.notes)}</td>
            <td data-col-group="tiers" class="num">${ratioCellHtml(f.t2.pass, f.t2.total, f.t2.notes)}</td>
            <td data-col-group="tiers" class="num">${ratioCellHtml(f.t3.pass, f.t3.total, f.t3.notes)}</td>
            <td data-col-group="tiers" class="num">${ratioCellHtml(f.t4.pass, f.t4.total, f.t4.notes)}</td>
            <td data-col-group="tiers" class="num">${ratioCellHtml(f.total_pass, f.total_req, f.totalNotes)}</td>
            <td data-col-group="full" class="num"><span class="badge badge-pass">${f.full_pass}</span></td>
            <td data-col-group="full" class="num"><span class="badge ${f.full_fail > 0 ? "badge-fail" : "badge-muted"}">${f.full_fail}</span></td>
            <td data-col-group="full" class="num">${f.full_not_measured}</td>
            <td data-col-group="full" class="num">${f.full_other}</td>
            <td data-col-group="stretch" class="num">${ratioCellHtml(f.stretch_pass, f.stretch_total, f.stretchNotes)}</td>
            <td data-col-group="full" class="tuning-cell">${escapeHtml(tuningText)}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderFamilyDetail() {
    const card = $("family-detail-card");
    const fam = state.families.find((f) => f.id === state.selectedFamilyId);
    if (!fam) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    $("fd-title").textContent = `${fam.label} (${fam.id})`;
    $("fd-subtitle").textContent = fam.selectedRecipe
      ? `Selected recipe: ${fam.selectedRecipe}`
      : "Selected family cohort";
    $("fd-page-link").href = githubBlobUrl(fam.pagePath);

    const heldSummary = fam.fullHeld
      ? `${fam.fullHeld.total}: ${fam.fullHeld.pass} PASS, ${fam.fullHeld.fail} FAIL, ${fam.fullHeld.not_measured} not measured, ${fam.fullHeld.other} other`
      : "—";
    const items = [
      ["Candidate ID", fam.candidateId || fam.selectedRecipe || "—"],
      ["Recipe SHA-256", fam.recipeHash || "—"],
      ["Source Digest", fam.sourceHash || "—"],
      ["Source Origin Commit", fam.sourceOriginCommit || "—"],
      ["Cohort Anchor", fam.cohortAnchor || "—"],
      ["Full-Suite (69)", `${fam.full_pass} PASS · ${fam.full_fail} FAIL · ${fam.full_not_measured} not measured · ${fam.full_other} other`],
      ["Held / Not Counted (in 69)", heldSummary],
      ["Qualification", fam.qualificationNote || fam.staleNote || "See family report"],
      ["Tuning Budget", fam.tuningBudget || "—"],
    ];
    if (fam.holdoutClaim) {
      items.push(["Selection vs Holdout", fam.holdoutClaim]);
    }
    if (fam.r256_joules_per_step != null || fam.ring16_j_to_pass != null || fam.grid100_j_to_pass != null) {
      items.push([
        "Matched Hardware Energy",
        [
          fam.r256_joules_per_step != null ? `r256 ${formatNum(fam.r256_joules_per_step, 2)} J/step` : null,
          fam.ring16_j_to_pass != null ? `ring16 ${formatJoules(fam.ring16_j_to_pass)}` : null,
          fam.grid100_j_to_pass != null ? `grid100 ${formatJoules(fam.grid100_j_to_pass)}` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      ]);
    }

    $("fd-body").innerHTML = items
      .map(
        ([label, val]) => `
        <div class="meta-box">
          <div class="meta-box-label">${escapeHtml(label)}</div>
          <div class="meta-box-value">${escapeHtml(val)}</div>
        </div>`
      )
      .join("");
  }

  /* ---------- Energy & J-to-PASS Efficiency ---------- */

  function getFilteredEnergyBenchmarks() {
    const q = state.energySearch.trim().toLowerCase();
    return state.energyBenchmarks.filter((row) => {
      if (state.energyFilter === "uncontended" && row.contended !== false) return false;
      if (state.energyFilter === "study" && !row.evidence_scope.includes("study")) return false;
      if (state.energyFilter === "receipt" && row.evidence_scope.includes("study")) return false;
      if (q) {
        const hay = `${row.candidate_id} ${row.family_id} ${row.optimizer_family} ${row.hardware_gpu} ${row.source_file} ${row.evidence_scope} ${row.source_commit}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function renderEnergyTable() {
    const table = $("energy-table");
    const tbody = $("energy-tbody");
    if (!table || !tbody) return;

    table.querySelectorAll("thead th.sortable").forEach((th) => {
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.energySort === state.energySort.key) {
        th.classList.add(state.energySort.dir === "asc" ? "sort-asc" : "sort-desc");
      }
    });

    const filtered = getFilteredEnergyBenchmarks();
    const numericKeys = new Set([
      "occam_pass",
      "r256_ms_per_step",
      "r256_joules_per_step",
      "r256_watts",
      "state_bytes_per_param",
      "ring16_j_to_pass",
      "grid100_j_to_pass",
    ]);

    const sorted = [...filtered].sort((a, b) => {
      const k = state.energySort.key;
      const dir = state.energySort.dir === "asc" ? 1 : -1;
      const va = a[k];
      const vb = b[k];
      if (numericKeys.has(k)) {
        const aMissing = va == null || !Number.isFinite(va);
        const bMissing = vb == null || !Number.isFinite(vb);
        if (aMissing && bMissing) return a.candidate_id.localeCompare(b.candidate_id);
        if (aMissing) return 1;
        if (bMissing) return -1;
        if (va !== vb) return (va - vb) * dir;
        return a.candidate_id.localeCompare(b.candidate_id);
      }
      const cmp = String(va ?? "").localeCompare(String(vb ?? "")) * dir;
      return cmp !== 0 ? cmp : a.candidate_id.localeCompare(b.candidate_id);
    });

    $("energy-count").textContent = `${sorted.length} / ${state.energyBenchmarks.length} benchmarks`;

    if (sorted.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="empty-cell">No hardware energy benchmark rows match the current filter.</td></tr>`;
      return;
    }

    tbody.innerHTML = sorted
      .map((row) => {
        const selectedCls = state.selectedEnergyKey === row.key ? "selected-row" : "";
        const suiteHtml =
          row.occam_pass != null && row.occam_executed != null
            ? ratioCellHtml(row.occam_pass, row.occam_executed)
            : `<span class="muted">—</span>`;
        const ringSub = row.ring16_first_pass_step != null ? `step ${row.ring16_first_pass_step}` : "";
        const gridSubs = [];
        if (row.grid100_first_pass_step != null) gridSubs.push(`step ${row.grid100_first_pass_step}`);
        if (row.grid100_j_to_sustained_pass != null) gridSubs.push(`sust ${formatJoules(row.grid100_j_to_sustained_pass)}`);
        const contendBadge =
          row.contended === false
            ? `<span class="badge badge-uncontended">uncontended</span>`
            : row.contended === true
            ? `<span class="badge badge-contended">contended</span>`
            : `<span class="badge badge-muted">unknown</span>`;
        const sourceLink = row.source_file
          ? `<a href="${escapeHtml(githubBlobUrl(row.source_file, row.source_commit || null))}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()"><code>${escapeHtml(row.source_file.split("/").pop())}</code></a>`
          : "—";
        const commitPill = row.source_commit ? `<code>${escapeHtml(shortHash(row.source_commit, 8))}</code>` : "";

        return `
          <tr class="clickable ${selectedCls}" data-energy-key="${escapeHtml(row.key)}">
            <td class="family-cell">
              <div class="family-name">${escapeHtml(row.candidate_id)}</div>
              <div class="family-meta">
                ${row.optimizer_family ? `<code>${escapeHtml(row.optimizer_family)}</code>` : ""}
                ${row.family_id ? `<span class="badge badge-muted">${escapeHtml(row.family_id)}</span>` : ""}
              </div>
            </td>
            <td class="num">${suiteHtml}</td>
            <td class="num">${row.r256_ms_per_step != null ? `${formatNum(row.r256_ms_per_step, 2)} ms` : "—"}</td>
            <td class="num"><strong>${row.r256_joules_per_step != null ? `${formatNum(row.r256_joules_per_step, 2)} J` : "—"}</strong></td>
            <td class="num">${row.r256_watts != null ? `${formatNum(row.r256_watts, 1)} W` : "—"}</td>
            <td class="num"><code>${escapeHtml(formatStateBpp(row.state_bytes_per_param))}</code></td>
            <td class="num">
              <div class="ratio-main">${escapeHtml(formatJoules(row.ring16_j_to_pass))}</div>
              ${ringSub ? `<span class="energy-sub">${escapeHtml(ringSub)}</span>` : ""}
            </td>
            <td class="num">
              <div class="ratio-main">${escapeHtml(formatJoules(row.grid100_j_to_pass))}</div>
              ${gridSubs.length ? `<span class="energy-sub">${escapeHtml(gridSubs.join(" · "))}</span>` : ""}
            </td>
            <td>
              <div>${escapeHtml(row.hardware_gpu || "—")}</div>
              <div style="margin-top:3px">${contendBadge}</div>
            </td>
            <td>
              <div><span class="badge badge-required">${escapeHtml(row.evidence_scope)}</span></div>
              <div class="family-meta">${sourceLink} ${commitPill}</div>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  function renderEnergyDetail() {
    const card = $("energy-detail-card");
    if (!card) return;
    const row = state.energyBenchmarks.find((item) => item.key === state.selectedEnergyKey);
    if (!row) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    $("ed-title").textContent = `${row.candidate_id} · Energy Telemetry`;
    $("ed-subtitle").textContent = `${row.optimizer_family || "—"} · ${row.hardware_gpu || "GPU unrecorded"} · ${row.evidence_scope}`;
    const srcBtn = $("ed-source-link");
    if (row.source_file) {
      srcBtn.href = githubBlobUrl(row.source_file, row.source_commit || null);
      srcBtn.classList.remove("hidden");
    } else {
      srcBtn.classList.add("hidden");
    }

    const items = [
      ["Candidate / Family", `${row.candidate_id}${row.family_id ? ` (${row.family_id})` : ""}`],
      ["Optimizer Family", row.optimizer_family || "—"],
      ["GPU Cohort", `${row.hardware_gpu || "—"} · ${row.contended === false ? "uncontended" : row.contended === true ? "contended" : "contention unknown"}`],
      ["r256 Step Cost", `${row.r256_ms_per_step != null ? `${formatNum(row.r256_ms_per_step, 2)} ms/step` : "—"} · ${row.r256_joules_per_step != null ? `${formatNum(row.r256_joules_per_step, 2)} J/step` : "—"} · ${row.r256_watts != null ? `${formatNum(row.r256_watts, 1)} W` : "—"}`],
      ["Optimizer State Footprint", row.state_bytes_per_param != null ? `${formatStateBpp(row.state_bytes_per_param)} B/param` : "—"],
      ["ring16 J-to-PASS", row.ring16_j_to_pass != null ? `${formatJoules(row.ring16_j_to_pass)} (first pass @ step ${row.ring16_first_pass_step ?? "—"})` : "—"],
      ["grid100 J-to-PASS (First)", row.grid100_j_to_pass != null ? `${formatJoules(row.grid100_j_to_pass)} (@ step ${row.grid100_first_pass_step ?? "—"})` : "—"],
      ["grid100 J-to-Sustained PASS", row.grid100_j_to_sustained_pass != null ? `${formatJoules(row.grid100_j_to_sustained_pass)} (@ step ${row.grid100_sustained_pass_step ?? "—"})` : "—"],
      ...(row.staggered100_j_to_pass != null
        ? [["staggered100 J-to-PASS", `${formatJoules(row.staggered100_j_to_pass)} (@ step ${row.staggered100_first_pass_step ?? "—"})`]]
        : []),
      ["Suite Pass (Occam)", row.occam_pass != null && row.occam_executed != null ? `${row.occam_pass} / ${row.occam_executed}` : "— (not in committed readout)"],
      ["Source Commit", row.source_commit || "—"],
      ["Source File", row.source_file || "—"],
      ["Evidence Scope", row.evidence_scope || "—"],
    ];

    $("ed-body").innerHTML = items
      .map(
        ([label, val]) => `
        <div class="meta-box">
          <div class="meta-box-label">${escapeHtml(label)}</div>
          <div class="meta-box-value">${escapeHtml(val)}</div>
        </div>`
      )
      .join("");
  }

  function populateFilterOptions() {
    const famSelect = $("filter-family");
    const prevFam = famSelect.value;
    famSelect.innerHTML = `<option value="">All families (${state.families.length})</option>` +
      state.families.map((f) => `<option value="${escapeHtml(f.id)}">${escapeHtml(f.label)}</option>`).join("");
    famSelect.value = prevFam;
  }

  function getFilteredReceipts() {
    const f = state.expFilters;
    const q = f.search.trim().toLowerCase();
    return state.receipts.filter((r) => {
      if (f.family && r.family_id !== f.family) return false;
      if (f.tier && String(r.tier) !== f.tier) return false;
      if (f.status && r.status !== f.status) return false;
      if (f.role && r.role !== f.role) return false;
      if (f.hasGif && !r.gif_path) return false;
      if (q) {
        const hay = `${r.task_id} ${r.family_id} ${r.family_label} ${r.candidate_id} ${r.status} ${r.role} ${r.reason} ${r.recipe_hash} ${r.source_hash} ${r.attempt_id}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function renderExplorer() {
    const table = $("explorer-table");
    const tbody = $("explorer-tbody");

    table.querySelectorAll("thead th.sortable").forEach((th) => {
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.expSort === state.expSort.key) {
        th.classList.add(state.expSort.dir === "asc" ? "sort-asc" : "sort-desc");
      }
    });

    const filtered = getFilteredReceipts();
    const sorted = [...filtered].sort((a, b) => {
      const k = state.expSort.key;
      const dir = state.expSort.dir === "asc" ? 1 : -1;
      const va = a[k];
      const vb = b[k];
      if (typeof va === "number" && typeof vb === "number") {
        if (va !== vb) return (va - vb) * dir;
        return a.task_id.localeCompare(b.task_id);
      }
      const cmp = String(va ?? "").localeCompare(String(vb ?? "")) * dir;
      return cmp !== 0 ? cmp : a.task_id.localeCompare(b.task_id);
    });

    $("explorer-count").textContent = `${sorted.length.toLocaleString()} / ${state.receipts.length.toLocaleString()} receipts`;

    if (sorted.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="empty-cell">No receipts match the current filters.</td></tr>`;
      $("explorer-page-info").textContent = "Showing 0 results";
      $("btn-show-more").classList.add("hidden");
      $("btn-show-all").classList.add("hidden");
      return;
    }

    const visible = sorted.slice(0, state.expLimit);
    tbody.innerHTML = visible
      .map((r) => {
        const selCls = state.selectedReceiptKey === r.key ? "selected-row" : "";
        const probUrl = `${githubBlobUrl("reports/forge/problems.md")}#${encodeURIComponent(r.task_id)}`;
        const provPills = [
          r.recipe_hash ? `rec:${shortHash(r.recipe_hash, 8)}` : "",
          r.source_hash ? `src:${shortHash(r.source_hash, 8)}` : "",
        ].filter(Boolean);
        const gifBtn = r.gif_path
          ? `<button type="button" class="btn btn-sm btn-primary" data-gif-jump="${escapeHtml(r.task_id)}" data-gif-fam="${escapeHtml(r.family_id)}">GIF</button>`
          : `<button type="button" class="btn btn-sm btn-ghost" data-gif-jump="${escapeHtml(r.task_id)}" data-gif-fam="${escapeHtml(r.family_id)}" title="Open in GIF Inspector">Inspect</button>`;

        return `
          <tr class="clickable ${selCls}" data-receipt-key="${escapeHtml(r.key)}">
            <td><a href="${escapeHtml(probUrl)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()"><code>${escapeHtml(r.task_id)}</code></a></td>
            <td><strong>${escapeHtml(r.family_label)}</strong></td>
            <td class="num">${r.tier > 0 ? `T${r.tier}` : "—"}</td>
            <td>${roleBadgeHtml(r.role)}</td>
            <td>${statusBadgeHtml(r.status)}</td>
            <td class="reason-cell">${escapeHtml(r.reason || "—")}</td>
            <td>${provPills.length ? `<code>${escapeHtml(provPills.join(" · "))}</code>` : "—"}</td>
            <td class="num">${gifBtn}</td>
          </tr>
        `;
      })
      .join("");

    $("explorer-page-info").textContent = `Showing ${visible.length.toLocaleString()} of ${sorted.length.toLocaleString()} matching receipts`;
    const remaining = sorted.length - visible.length;
    $("btn-show-more").classList.toggle("hidden", remaining <= 0);
    $("btn-show-all").classList.toggle("hidden", remaining <= 0);
  }

  function renderReceiptDetail() {
    const card = $("receipt-detail-card");
    const r = state.receipts.find((item) => item.key === state.selectedReceiptKey);
    if (!r) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    $("rd-title").textContent = `${r.task_id} · ${r.family_label}`;
    $("rd-subtitle").textContent = `Tier ${r.tier || "—"} · ${r.role} · ${r.status}`;

    const def = state.taskDefinitions.get(r.task_id);
    const gifUrl = r.gif_path ? hfResolveUrl(r.gif_path) : null;
    const metaPairs = [
      ["Status", r.recorded_status && r.recorded_status !== r.status ? `${r.status} (recorded: ${r.recorded_status})` : r.status],
      ["Role / Tier", `${r.role} · Tier ${r.tier || "—"}`],
      ["Candidate ID", r.candidate_id || "—"],
      ["Attempt ID", r.attempt_id || "—"],
      ["Recipe Hash", r.recipe_hash || "—"],
      ["Source Hash", r.source_hash || "—"],
      ["Views", r.views.length > 0 ? r.views.join(", ") : "—"],
      ["Wall Time", r.wall_seconds != null ? `${r.wall_seconds.toFixed(2)} s` : "—"],
      ["Reason / Coverage", r.reason || (def && def.summary) || "—"],
    ];
    if (r.energy) {
      const e = r.energy;
      const j2pParts = [];
      if (e.joules_to_first_pass != null) {
        j2pParts.push(`First: ${formatJoules(e.joules_to_first_pass)}${e.first_pass_step != null ? ` (@ step ${e.first_pass_step})` : ""}`);
      }
      if (e.joules_to_sustained_pass != null) {
        j2pParts.push(`Sustained: ${formatJoules(e.joules_to_sustained_pass)}${e.sustained_pass_step != null ? ` (@ step ${e.sustained_pass_step})` : ""}`);
      }
      metaPairs.push(["J-to-PASS", j2pParts.length > 0 ? j2pParts.join(" · ") : "—"]);
      metaPairs.push(["Energy / Step", e.joules_per_step != null ? `${formatNum(e.joules_per_step, 3)} J/step` : "—"]);
      metaPairs.push(["Avg Power", e.avg_power_watts != null ? `${formatNum(e.avg_power_watts, 1)} W` : "—"]);
      metaPairs.push([
        "Total Energy / Backend",
        `${e.joules_total != null ? formatJoules(e.joules_total) : "—"}${e.energy_backend ? ` · ${e.energy_backend}` : ""}`,
      ]);
    } else {
      metaPairs.push(["Hardware Energy (J-to-PASS)", "— (not measured)"]);
    }
    if (def && def.re_entry_rule) {
      metaPairs.push(["Re-entry Rule", def.re_entry_rule]);
    }

    let html = `<div class="detail-meta-grid">` +
      metaPairs
        .map(
          ([label, val]) => `
          <div class="meta-box">
            <div class="meta-box-label">${escapeHtml(label)}</div>
            <div class="meta-box-value">${escapeHtml(val)}</div>
          </div>`
        )
        .join("") +
      `</div>`;

    if (gifUrl) {
      html += `<div style="margin-top:10px"><a href="${escapeHtml(gifUrl)}" target="_blank" rel="noopener noreferrer">Open training GIF on Hugging Face ↗</a></div>`;
    }
    if (r.metrics && Object.keys(r.metrics).length > 0) {
      html += `<pre class="metrics-pre">${escapeHtml(JSON.stringify(r.metrics, null, 2))}</pre>`;
    }
    $("rd-body").innerHTML = html;
  }

  /* ---------- GIF Inspector ---------- */

  function populateGifSelectors() {
    const taskMap = new Map();
    for (const r of state.receipts) {
      if (!taskMap.has(r.task_id)) {
        taskMap.set(r.task_id, { taskId: r.task_id, tier: r.tier, role: r.role, gifCount: 0 });
      }
      if (r.gif_path) taskMap.get(r.task_id).gifCount += 1;
    }

    let tasks = [...taskMap.values()].sort((a, b) => {
      if (b.gifCount !== a.gifCount) return b.gifCount - a.gifCount;
      if (a.tier !== b.tier) return a.tier - b.tier;
      return a.taskId.localeCompare(b.taskId);
    });

    if (state.gifOnlyAvailable) {
      tasks = tasks.filter((t) => t.gifCount > 0);
    }

    const taskSelect = $("gif-task");
    if (tasks.length === 0) {
      taskSelect.innerHTML = `<option value="">No tasks available</option>`;
      state.gifTask = "";
    } else {
      taskSelect.innerHTML = tasks
        .map((t) => {
          const gifTag = t.gifCount > 0 ? ` [${t.gifCount} GIF${t.gifCount === 1 ? "" : "s"}]` : " [no GIF yet]";
          const tierTag = t.tier > 0 ? `T${t.tier} · ` : "";
          return `<option value="${escapeHtml(t.taskId)}">${escapeHtml(tierTag + t.taskId + gifTag)}</option>`;
        })
        .join("");
      if (!tasks.some((t) => t.taskId === state.gifTask)) {
        state.gifTask = tasks[0].taskId;
      }
      taskSelect.value = state.gifTask;
    }

    const famOptions = state.families
      .map((f) => `<option value="${escapeHtml(f.id)}">${escapeHtml(f.label)}</option>`)
      .join("");
    $("gif-family-left").innerHTML = famOptions;
    $("gif-family-right").innerHTML = famOptions;

    if (!state.families.some((f) => f.id === state.gifLeft) && state.families.length > 0) {
      // Pick first family with a GIF for current task if possible
      const withGif = state.receipts.filter((r) => r.task_id === state.gifTask && r.gif_path);
      state.gifLeft = (withGif[0] && withGif[0].family_id) || state.families[0].id;
    }
    if (!state.families.some((f) => f.id === state.gifRight) && state.families.length > 0) {
      const withGif = state.receipts.filter((r) => r.task_id === state.gifTask && r.gif_path && r.family_id !== state.gifLeft);
      state.gifRight =
        (withGif[0] && withGif[0].family_id) ||
        (state.families.find((f) => f.id !== state.gifLeft) || state.families[0]).id;
    }
    $("gif-family-left").value = state.gifLeft;
    $("gif-family-right").value = state.gifRight;
  }

  function renderGifCard(family, receipt) {
    const statusHtml = receipt ? statusBadgeHtml(receipt.status) : `<span class="badge badge-not_run">not_run</span>`;
    const gifUrl = receipt && receipt.gif_path ? hfResolveUrl(receipt.gif_path) : null;
    const famPage = githubBlobUrl(family.pagePath);
    const hashText = (receipt && receipt.source_hash) || family.sourceHash;

    const mediaHtml = gifUrl
      ? `<img src="${escapeHtml(gifUrl)}" alt="${escapeHtml(family.label)} training GIF for ${escapeHtml(state.gifTask)}" loading="lazy" data-gif-img="1">`
      : `<div class="gif-placeholder">
           <strong>No training GIF published yet</strong>
           <span>Recorded status for <code>${escapeHtml(state.gifTask)}</code>: ${escapeHtml(receipt ? receipt.status : "not_run")}.</span>
           ${receipt && receipt.reason ? `<div style="margin-top:6px">${escapeHtml(receipt.reason)}</div>` : ""}
         </div>`;

    return `
      <article class="gif-card">
        <div class="gif-card-head">
          <div>
            <div class="gif-card-title">${escapeHtml(family.label)}</div>
            <div class="gif-card-sub">${escapeHtml(family.selectedRecipe || family.candidateId || family.id)}</div>
          </div>
          ${statusHtml}
        </div>
        <div class="gif-media-box">${mediaHtml}</div>
        <div class="gif-card-foot">
          <span>${hashText ? `src: <code>${escapeHtml(shortHash(hashText, 10))}</code>` : "Recorded cohort"}</span>
          <span>
            ${gifUrl ? `<a href="${escapeHtml(gifUrl)}" target="_blank" rel="noopener noreferrer">Raw GIF ↗</a> · ` : ""}
            <a href="${escapeHtml(famPage)}" target="_blank" rel="noopener noreferrer">Family report ↗</a>
          </span>
        </div>
      </article>
    `;
  }

  function renderGifInspector() {
    const stage = $("gif-stage");
    const compareControls = state.gifLayout === "compare";
    $("gif-left-group").classList.toggle("hidden", !compareControls);
    $("gif-right-group").classList.toggle("hidden", !compareControls);
    $("gif-swap-group").classList.toggle("hidden", !compareControls);
    stage.className = compareControls ? "gif-stage gif-stage-compare" : "gif-stage gif-stage-grid";

    if (!state.gifTask) {
      $("gif-task-meta").textContent = "";
      stage.innerHTML = `<div class="empty-cell">No task selected.</div>`;
      return;
    }

    const byFamily = new Map();
    for (const r of state.receipts) {
      if (r.task_id === state.gifTask) byFamily.set(r.family_id, r);
    }

    const def = state.taskDefinitions.get(state.gifTask);
    const gifAvail = [...byFamily.values()].filter((r) => Boolean(r.gif_path)).length;
    const probLink = `${githubBlobUrl("reports/forge/problems.md")}#${encodeURIComponent(state.gifTask)}`;
    $("gif-task-meta").innerHTML = `
      <strong><code>${escapeHtml(state.gifTask)}</code></strong> ·
      ${gifAvail} of ${state.families.length} families have a published GIF ·
      <a href="${escapeHtml(probLink)}" target="_blank" rel="noopener noreferrer">Problem specification ↗</a>
      ${def && def.summary ? `<div style="margin-top:4px">${escapeHtml(def.summary)}</div>` : ""}
    `;

    let selectedFamilies = [];
    if (compareControls) {
      const leftFam = state.families.find((f) => f.id === state.gifLeft);
      const rightFam = state.families.find((f) => f.id === state.gifRight);
      if (leftFam) selectedFamilies.push(leftFam);
      if (rightFam) selectedFamilies.push(rightFam);
    } else {
      selectedFamilies = [...state.families].sort((a, b) => {
        const ga = byFamily.get(a.id)?.gif_path ? 1 : 0;
        const gb = byFamily.get(b.id)?.gif_path ? 1 : 0;
        return gb - ga || a.label.localeCompare(b.label);
      });
    }

    stage.innerHTML = selectedFamilies.map((f) => renderGifCard(f, byFamily.get(f.id))).join("");

    // Graceful fallback if a GIF URL returns 404 or fails to load
    stage.querySelectorAll("img[data-gif-img]").forEach((img) => {
      img.addEventListener("error", () => {
        const box = img.parentElement;
        if (!box) return;
        box.innerHTML = `
          <div class="gif-placeholder">
            <strong>GIF not yet available at this revision</strong>
            <span>Checked <code>${escapeHtml(img.getAttribute("src") || "")}</code></span>
          </div>`;
      });
    });
  }

  function renderProvenanceSection() {
    const m = state.meta;
    const dsViewer = `https://huggingface.co/datasets/${m.datasetRepo}`;
    const rawJson = hfResolveUrl(CONFIG.dataPath, m.hfRevision);
    const declaredViewsStr = Object.entries(m.declaredViewRevisions || {})
      .map(([k, v]) => `${k} rev ${v}`)
      .join(", ");
    const cards = [
      ["GitHub Repository", `<a href="https://github.com/${GITHUB_REPO}" target="_blank" rel="noopener noreferrer">github.com/${GITHUB_REPO} ↗</a>`],
      ["Authoritative Leaderboard", `<a href="${escapeHtml(githubBlobUrl("reports/forge/technique-inventory.md"))}" target="_blank" rel="noopener noreferrer">reports/forge/technique-inventory.md ↗</a>`],
      ["HF Dataset Viewer", `<a href="${escapeHtml(dsViewer)}" target="_blank" rel="noopener noreferrer">${escapeHtml(m.datasetRepo)} ↗</a>`],
      ["Live JSON Source", `<a href="${escapeHtml(rawJson)}" target="_blank" rel="noopener noreferrer">${escapeHtml(CONFIG.dataPath)} ↗</a>`],
      ["Pinned HF Revision", `<code>${escapeHtml(m.hfRevision || "main")}</code>`],
      ["Git Source Commit", `<code>${escapeHtml(m.gitCommit || "develop")}</code>`],
      ["Full-Suite List", `<code>${escapeHtml(m.fullSuiteList || "—")}</code>${m.fullSuiteSha256 ? ` · <code>${escapeHtml(shortHash(m.fullSuiteSha256, 12))}</code>` : ""}`],
      ["Declared View Revisions", declaredViewsStr ? `<code>${escapeHtml(declaredViewsStr)}</code>` : `<code>${escapeHtml(m.inputDigest || "—")}</code>`],
    ];

    $("provenance-grid").innerHTML = cards
      .map(
        ([label, html]) => `
        <div class="meta-box">
          <div class="meta-box-label">${escapeHtml(label)}</div>
          <div class="meta-box-value">${html}</div>
        </div>`
      )
      .join("");
  }

  function renderAll() {
    renderProvenanceBar();
    renderKpis();
    renderLeaderboard();
    renderFamilyDetail();
    renderEnergyTable();
    renderEnergyDetail();
    populateFilterOptions();
    renderExplorer();
    renderReceiptDetail();
    populateGifSelectors();
    renderGifInspector();
    renderProvenanceSection();
  }

  /* ---------- Data Loading ---------- */

  async function fetchHfDatasetSha() {
    try {
      const resp = await fetch(`https://huggingface.co/api/datasets/${CONFIG.datasetRepo}`, {
        headers: { Accept: "application/json" },
      });
      if (!resp.ok) return null;
      const info = await resp.json();
      return info.sha || null;
    } catch {
      return null;
    }
  }

  async function loadData() {
    setStatus(`Loading live leaderboard from ${CONFIG.datasetRepo} (${CONFIG.requestedRev})…`, "loading");
    try {
      const [hfSha, dataResp] = await Promise.all([
        fetchHfDatasetSha(),
        fetch(
          `https://huggingface.co/datasets/${CONFIG.datasetRepo}/resolve/${encodeURIComponent(CONFIG.requestedRev)}/${CONFIG.dataPath}`,
          { headers: { Accept: "application/json" }, cache: "no-cache" }
        ),
      ]);

      if (!dataResp.ok) {
        throw new Error(`HTTP ${dataResp.status} fetching ${CONFIG.dataPath} from ${CONFIG.datasetRepo}`);
      }

      const raw = await dataResp.json();
      normalizeDataset(raw, hfSha);
      setStatus(null);
      renderAll();
    } catch (err) {
      setStatus(
        `Unable to load live data from ${CONFIG.datasetRepo}/${CONFIG.dataPath}: ${err.message}. Check that data/leaderboard.json is published on Hugging Face or retry.`,
        "error"
      );
      renderProvenanceBar();
      renderProvenanceSection();
    }
  }

  /* ---------- Theme & Event Wiring ---------- */

  const THEMES = ["auto", "dark", "light"];
  const THEME_LABELS = { auto: "◐ Auto", dark: "☾ Dark", light: "☀ Light" };

  function applyTheme(theme) {
    const next = THEMES.includes(theme) ? theme : "auto";
    document.documentElement.dataset.theme = next;
    $("btn-theme").textContent = THEME_LABELS[next];
    try {
      localStorage.setItem("particlegan_lb_theme", next);
    } catch {
      // ignore storage restrictions inside sandboxed iframes
    }
  }

  function wireEvents() {
    // Theme toggle
    let savedTheme = "auto";
    try {
      savedTheme = localStorage.getItem("particlegan_lb_theme") || "auto";
    } catch {
      savedTheme = "auto";
    }
    applyTheme(savedTheme);

    $("btn-theme").addEventListener("click", () => {
      const current = document.documentElement.dataset.theme || "auto";
      const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
      applyTheme(next);
    });

    $("btn-refresh").addEventListener("click", () => loadData());

    // Leaderboard column modes
    document.querySelectorAll("[data-lb-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-lb-mode]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.lbMode = btn.dataset.lbMode;
        renderLeaderboard();
      });
    });

    // Leaderboard sorting
    $("leaderboard-table").querySelectorAll("thead th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        if (state.lbSort.key === key) {
          state.lbSort.dir = state.lbSort.dir === "asc" ? "desc" : "asc";
        } else {
          state.lbSort = { key, dir: key === "label" || key === "tuning" ? "asc" : "desc" };
        }
        renderLeaderboard();
      });
    });

    // Leaderboard row selection
    $("leaderboard-tbody").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-family-id]");
      if (!tr) return;
      const id = tr.dataset.familyId;
      state.selectedFamilyId = state.selectedFamilyId === id ? null : id;
      renderLeaderboard();
      renderFamilyDetail();
    });

    $("fd-close-btn").addEventListener("click", () => {
      state.selectedFamilyId = null;
      renderLeaderboard();
      renderFamilyDetail();
    });

    $("fd-filter-btn").addEventListener("click", () => {
      if (!state.selectedFamilyId) return;
      state.expFilters.family = state.selectedFamilyId;
      $("filter-family").value = state.selectedFamilyId;
      state.expLimit = 100;
      renderExplorer();
      $("explorer-section").scrollIntoView({ behavior: "smooth" });
    });

    // Energy & J-to-PASS controls
    document.querySelectorAll("[data-energy-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-energy-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.energyFilter = btn.dataset.energyFilter;
        renderEnergyTable();
      });
    });

    const energySearch = $("energy-search");
    if (energySearch) {
      energySearch.addEventListener("input", (e) => {
        state.energySearch = e.target.value;
        renderEnergyTable();
      });
    }

    const energyTable = $("energy-table");
    if (energyTable) {
      energyTable.querySelectorAll("thead th.sortable").forEach((th) => {
        th.addEventListener("click", () => {
          const key = th.dataset.energySort;
          if (state.energySort.key === key) {
            state.energySort.dir = state.energySort.dir === "asc" ? "desc" : "asc";
          } else {
            state.energySort = { key, dir: key === "occam_pass" ? "desc" : "asc" };
          }
          renderEnergyTable();
        });
      });
    }

    const energyTbody = $("energy-tbody");
    if (energyTbody) {
      energyTbody.addEventListener("click", (e) => {
        const tr = e.target.closest("tr[data-energy-key]");
        if (!tr) return;
        const key = tr.dataset.energyKey;
        state.selectedEnergyKey = state.selectedEnergyKey === key ? null : key;
        renderEnergyTable();
        renderEnergyDetail();
      });
    }

    const edClose = $("ed-close-btn");
    if (edClose) {
      edClose.addEventListener("click", () => {
        state.selectedEnergyKey = null;
        renderEnergyTable();
        renderEnergyDetail();
      });
    }

    // Explorer filters
    $("filter-family").addEventListener("change", (e) => {
      state.expFilters.family = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-tier").addEventListener("change", (e) => {
      state.expFilters.tier = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-status").addEventListener("change", (e) => {
      state.expFilters.status = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-role").addEventListener("change", (e) => {
      state.expFilters.role = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-search").addEventListener("input", (e) => {
      state.expFilters.search = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-has-gif").addEventListener("change", (e) => {
      state.expFilters.hasGif = e.target.checked;
      state.expLimit = 100;
      renderExplorer();
    });
    $("btn-reset-filters").addEventListener("click", () => {
      state.expFilters = { family: "", tier: "", status: "", role: "", search: "", hasGif: false };
      $("filter-family").value = "";
      $("filter-tier").value = "";
      $("filter-status").value = "";
      $("filter-role").value = "";
      $("filter-search").value = "";
      $("filter-has-gif").checked = false;
      state.expLimit = 100;
      renderExplorer();
    });

    // Explorer sorting
    $("explorer-table").querySelectorAll("thead th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.expSort;
        if (state.expSort.key === key) {
          state.expSort.dir = state.expSort.dir === "asc" ? "desc" : "asc";
        } else {
          state.expSort = { key, dir: "asc" };
        }
        renderExplorer();
      });
    });

    // Explorer row click + GIF jump button
    $("explorer-tbody").addEventListener("click", (e) => {
      const jumpBtn = e.target.closest("button[data-gif-jump]");
      if (jumpBtn) {
        e.stopPropagation();
        state.gifTask = jumpBtn.dataset.gifJump;
        state.gifLeft = jumpBtn.dataset.gifFam;
        state.gifOnlyAvailable = false;
        $("gif-only-available").checked = false;
        populateGifSelectors();
        renderGifInspector();
        $("gif-section").scrollIntoView({ behavior: "smooth" });
        return;
      }
      const tr = e.target.closest("tr[data-receipt-key]");
      if (!tr) return;
      const key = tr.dataset.receiptKey;
      state.selectedReceiptKey = state.selectedReceiptKey === key ? null : key;
      renderExplorer();
      renderReceiptDetail();
    });

    $("btn-show-more").addEventListener("click", () => {
      state.expLimit += 100;
      renderExplorer();
    });
    $("btn-show-all").addEventListener("click", () => {
      state.expLimit = state.receipts.length + 1;
      renderExplorer();
    });

    $("rd-close-btn").addEventListener("click", () => {
      state.selectedReceiptKey = null;
      renderExplorer();
      renderReceiptDetail();
    });

    $("rd-open-gif-btn").addEventListener("click", () => {
      const r = state.receipts.find((item) => item.key === state.selectedReceiptKey);
      if (!r) return;
      state.gifTask = r.task_id;
      state.gifLeft = r.family_id;
      state.gifOnlyAvailable = false;
      $("gif-only-available").checked = false;
      populateGifSelectors();
      renderGifInspector();
      $("gif-section").scrollIntoView({ behavior: "smooth" });
    });

    // GIF Inspector controls
    document.querySelectorAll("[data-gif-layout]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-gif-layout]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.gifLayout = btn.dataset.gifLayout;
        renderGifInspector();
      });
    });

    $("gif-task").addEventListener("change", (e) => {
      state.gifTask = e.target.value;
      renderGifInspector();
    });
    $("gif-family-left").addEventListener("change", (e) => {
      state.gifLeft = e.target.value;
      renderGifInspector();
    });
    $("gif-family-right").addEventListener("change", (e) => {
      state.gifRight = e.target.value;
      renderGifInspector();
    });
    $("btn-swap-gif").addEventListener("click", () => {
      const tmp = state.gifLeft;
      state.gifLeft = state.gifRight;
      state.gifRight = tmp;
      $("gif-family-left").value = state.gifLeft;
      $("gif-family-right").value = state.gifRight;
      renderGifInspector();
    });
    $("gif-only-available").addEventListener("change", (e) => {
      state.gifOnlyAvailable = e.target.checked;
      populateGifSelectors();
      renderGifInspector();
    });
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => {
      wireEvents();
      loadData();
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      state,
      normalizeDataset,
      normalizeFamily,
      normalizeReceipt,
      normalizeEnergyBenchmark,
      normalizeEnergyObj,
      normalizeStatus,
      normalizeRole,
      normalizeGifPath,
      hfResolveUrl,
      formatJoules,
    };
  }
})();
