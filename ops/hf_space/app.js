/* ParticleGAN Forge Leaderboard — Flagship Static Space App
 * Loads live data & training GIFs directly from Hugging Face Dataset:
 * https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence
 * Also supports local preview via ?local=1&data_path=data/leaderboard.json or ?dataset=local
 */

(() => {
  "use strict";

  const DEFAULT_DATASET_REPO = "HyperGAN/particlegan-forge-evidence";
  const DEFAULT_PAGES_BASE_URL = "https://hypergan.github.io/ParticleGAN";
  const GITHUB_REPO = "HyperGAN/ParticleGAN";
  const DEFAULT_BRANCH = "develop";

  const DEFAULT_EPISTEMIC_EPIGRAPH =
    "Our goal is not to preserve any fixed set of laws. Our goal is to discover the laws of learning as evidence advances.";
  const DEFAULT_EPISTEMIC_NOTE =
    "This law taxonomy is a living scientific framework derived from current mathematical and empirical audits. Individual laws may be refined, split, merged, or replaced as new counterexamples and evidence emerge.";

  const params =
    typeof window !== "undefined" && window.location
      ? new URLSearchParams(window.location.search)
      : new URLSearchParams("");

  const CONFIG = {
    datasetRepo: params.get("dataset") || DEFAULT_DATASET_REPO,
    requestedRev: params.get("rev") || "main",
    mediaRev: params.get("media_rev") || null,
    dataPath: params.get("data_path") || "data/leaderboard.json",
    localMode: params.get("local") === "1" || params.get("local") === "true",
  };

  const state = {
    raw: null,
    meta: {
      datasetRepo: CONFIG.datasetRepo,
      pagesBaseUrl: DEFAULT_PAGES_BASE_URL,
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
    lawsMeta: {
      schema_version: 1,
      epistemic_status: "provisional_evolving",
      epigraph: DEFAULT_EPISTEMIC_EPIGRAPH,
      epistemic_note: DEFAULT_EPISTEMIC_NOTE,
      doc_path: "docs/four-laws.md",
      pages_url: `${DEFAULT_PAGES_BASE_URL}/laws/`,
      source: "unavailable",
    },
    laws: [],
    lawLookup: new Map(),
    families: [],
    receipts: [],
    energyBenchmarks: [],
    taskDefinitions: new Map(),
    fourLawsL4: null,
    fourLawsV2L4: null,
    fourLawsVersion: "auto",
    selectedFamilyId: null,
    selectedFormulationFamilyId: null,
    selectedLawCell: null, // { familyId, lawId }
    selectedReceiptKey: null,
    selectedEnergyKey: null,
    activeExperimentId: null,
    expModalGifFamilies: [],
    lbMode: "all",
    lbSort: { key: "total_pass", dir: "desc" },
    energySort: { key: "grid100_j_to_pass", dir: "asc" },
    energyFilter: "all",
    energySearch: "",
    paretoX: "grid100_j_to_pass",
    paretoY: "occam",
    expSort: { key: "tier", dir: "asc" },
    expFilters: {
      family: "",
      law: "",
      edgeCase: "",
      tier: "",
      status: "",
      role: "",
      search: "",
      hasGif: false,
    },
    expLimit: 100,
    gifLayout: "compare",
    gifLawFilter: "",
    gifSearch: "",
    gifTask: "",
    gifLeft: "",
    gifRight: "",
    gifOnlyAvailable: true,
    gifHideMissingInGrid: false,
    gifSyncNonce: 0,
    gifNextSlot: "right",
    gifLightbox: null, // { taskId, familyIds, index }
  };

  let mermaidRenderSeq = 0;

  /* ---------- Utilities ---------- */

  const $ = (id) => (typeof document !== "undefined" ? document.getElementById(id) : null);

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

  function isLocalMode(cfg = CONFIG) {
    if (!cfg) return false;
    if (cfg.localMode) return true;
    if (cfg.datasetRepo === "local") return true;
    const dp = String(cfg.dataPath || "");
    return dp.startsWith("./") || dp.startsWith("/") || /^https?:\/\//i.test(dp);
  }

  function resolveDataUrl(cfg = CONFIG) {
    const dp = String((cfg && cfg.dataPath) || "data/leaderboard.json");
    if (isLocalMode(cfg)) return dp;
    const repo = (cfg && cfg.datasetRepo) || DEFAULT_DATASET_REPO;
    const rev = (cfg && cfg.requestedRev) || "main";
    return `https://huggingface.co/datasets/${repo}/resolve/${encodeURIComponent(rev)}/${dp.replace(/^\/+/, "")}`;
  }

  function hfResolveUrl(path, revOverride = null) {
    if (!path) return null;
    if (/^https?:\/\//i.test(path)) return path;
    const cleaned = String(path).replace(/^\/+/, "");
    if (state.meta.datasetRepo === "local") {
      return `./${cleaned}`;
    }
    const rev = revOverride || state.meta.mediaRevision || state.meta.hfRevision || "main";
    return `https://huggingface.co/datasets/${state.meta.datasetRepo}/resolve/${encodeURIComponent(rev)}/${cleaned}`;
  }

  function githubBlobUrl(relPath, refOverride = null) {
    const ref = refOverride || state.meta.gitCommit || DEFAULT_BRANCH;
    const cleaned = String(relPath || "").replace(/^\/+/, "");
    return `https://github.com/${GITHUB_REPO}/blob/${encodeURIComponent(ref)}/${cleaned}`;
  }

  function pagesUrl(subPath = "") {
    const base = (state.meta.pagesBaseUrl || DEFAULT_PAGES_BASE_URL).replace(/\/+$/, "");
    const cleaned = String(subPath || "").replace(/^\/+/, "");
    return cleaned ? `${base}/${cleaned}` : `${base}/`;
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
    const formatted =
      value >= 1000
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
    if (
      lower === "not_run" ||
      lower === "not run" ||
      lower === "not_measured" ||
      lower === "not measured" ||
      lower === "unknown" ||
      lower === "unmeasured" ||
      lower === ""
    ) {
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
    if (s.startsWith("/")) return null;
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

  function lawStatusMeta(rawStatus) {
    const s = String(rawStatus || "unrecorded").trim().toLowerCase();
    if (s === "addressed") return { key: "addressed", icon: "✓", label: "Addressed", badgeCls: "badge-law-addressed" };
    if (s === "partial") return { key: "partial", icon: "◐", label: "Partial", badgeCls: "badge-law-partial" };
    if (s === "unaddressed") return { key: "unaddressed", icon: "○", label: "Unaddressed", badgeCls: "badge-law-unaddressed" };
    if (s === "not_applicable") return { key: "not_applicable", icon: "—", label: "N/A", badgeCls: "badge-law-na" };
    return { key: "unrecorded", icon: "·", label: "Unrecorded", badgeCls: "badge-law-na" };
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

  function isEffectiveDarkTheme() {
    if (typeof document === "undefined") return false;
    const theme = document.documentElement.dataset.theme || "auto";
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function renderLatexHtml(latex, displayMode = true) {
    const src = String(latex || "").trim();
    if (!src) return "";
    if (typeof window !== "undefined" && window.katex && typeof window.katex.renderToString === "function") {
      try {
        return window.katex.renderToString(src, {
          displayMode,
          throwOnError: false,
          strict: false,
        });
      } catch {
        // fall through to monospace code fallback
      }
    }
    return `<code class="math-fallback">${escapeHtml(src)}</code>`;
  }

  function renderInlineMathHtml(text) {
    const raw = String(text || "");
    if (!raw || !raw.includes("$")) return escapeHtml(raw);
    const parts = [];
    const tokenRe = /\$`([^`\n]+)`\$|\$([^\s$](?:[^$\n]*[^\s$])?)\$/g;
    let last = 0;
    let m;
    while ((m = tokenRe.exec(raw)) !== null) {
      if (m.index > last) {
        parts.push(escapeHtml(raw.slice(last, m.index)));
      }
      const expr = m[1] !== undefined ? m[1] : m[2];
      parts.push(renderLatexHtml(expr, false));
      last = tokenRe.lastIndex;
    }
    if (last < raw.length) {
      parts.push(escapeHtml(raw.slice(last)));
    }
    return parts.join("");
  }

  function renderMermaidBlocks(container) {
    if (!container || typeof window === "undefined") return;
    const hosts = container.querySelectorAll("[data-mermaid-source]");
    if (!hosts.length) return;
    const mermaid = window.mermaid;
    if (!mermaid || typeof mermaid.render !== "function") return;
    const dark = isEffectiveDarkTheme();
    try {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: dark ? "dark" : "neutral",
        themeVariables: dark
          ? {
              background: "#0d1117",
              primaryColor: "#1e293b",
              primaryTextColor: "#f1f5f9",
              primaryBorderColor: "#818cf8",
              lineColor: "#94a3b8",
              secondaryColor: "#172033",
              tertiaryColor: "#111827",
              edgeLabelBackground: "#0f172a",
              textColor: "#f1f5f9",
              nodeTextColor: "#f1f5f9",
              fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
            }
          : {
              background: "#f8fafc",
              primaryColor: "#eef2ff",
              primaryTextColor: "#0f172a",
              primaryBorderColor: "#4338ca",
              lineColor: "#475569",
              secondaryColor: "#e0f2fe",
              tertiaryColor: "#ffffff",
              edgeLabelBackground: "#ffffff",
              textColor: "#0f172a",
              nodeTextColor: "#0f172a",
              fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
            },
        flowchart: { useMaxWidth: true, htmlLabels: true, curve: "basis" },
      });
    } catch {
      return;
    }

    hosts.forEach((host) => {
      const code = host.dataset.mermaidSource || "";
      if (!code.trim()) return;
      const renderId = `mmd-${++mermaidRenderSeq}`;
      Promise.resolve()
        .then(() => mermaid.render(renderId, code))
        .then((res) => {
          if (res && typeof res.svg === "string" && host.isConnected) {
            host.innerHTML = res.svg;
          }
        })
        .catch(() => {
          // Leave plain-text Mermaid fallback intact
        });
    });
  }

  function sanitizeInlineSvg(svgStr) {
    if (typeof svgStr !== "string") return null;
    const cleaned = svgStr.replace(/^<\?xml[^>]*\?>\s*/i, "").trim();
    if (!cleaned.startsWith("<svg")) return null;
    if (/<script\b|javascript:/i.test(cleaned)) return null;
    return cleaned;
  }

  function copyTextWithFeedback(text, buttonEl) {
    const done = () => {
      if (!buttonEl) return;
      const prev = buttonEl.textContent;
      buttonEl.textContent = "Copied ✓";
      setTimeout(() => {
        if (buttonEl.isConnected) buttonEl.textContent = prev;
      }, 1600);
    };
    if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => {});
    }
  }

  /* ---------- Schema Normalization ---------- */

  function normalizeLaw(rawLaw) {
    if (!rawLaw || typeof rawLaw !== "object") return null;
    const id = String(rawLaw.id || "").trim();
    if (!id) return null;
    const subLaws = Array.isArray(rawLaw.sub_laws)
      ? rawLaw.sub_laws
          .filter((s) => s && typeof s === "object" && s.id)
          .map((s) => ({
            id: String(s.id).trim(),
            short_label: String(s.short_label || s.id).trim(),
            title: String(s.title || s.id).trim(),
            doc_anchor: String(s.doc_anchor || rawLaw.doc_anchor || "").trim(),
          }))
      : [];
    return {
      id,
      short_label: String(rawLaw.short_label || id).trim(),
      title: String(rawLaw.title || id).trim(),
      domain_level: String(rawLaw.domain_level || rawLaw.title || id).trim(),
      question: String(rawLaw.question || "").trim(),
      doc_anchor: String(rawLaw.doc_anchor || "").trim(),
      status: String(rawLaw.status || "active").trim(),
      sub_laws: subLaws,
    };
  }

  function extractTierStats(fam, tierNum) {
    const key = String(tierNum);
    const tierObj = (fam.tiers && (fam.tiers[key] || fam.tiers[`tier_${key}`] || fam.tiers[`tier${key}`])) || {};
    const counts = tierObj.counts || {};
    const pass = toInt(
      tierObj.passed ?? tierObj.pass ?? counts.PASS ?? fam[`tier${key}_pass`] ?? fam[`tier_${key}_pass`],
      0
    );
    const total = toInt(tierObj.total ?? fam[`tier${key}_total`] ?? fam[`tier_${key}_total`], 0);
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
    const totalDiag = toInt(
      totalObj.diagnostic ?? totalObj.diagnostic_attempts ?? rawFam.total_diagnostic,
      t1.diagnostic + t2.diagnostic + t3.diagnostic + t4.diagnostic
    );
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
    const famPagesUrl = rawFam.pages_url || pagesUrl(`families/${id}/`);
    const lawMechanisms =
      rawFam.law_mechanisms && typeof rawFam.law_mechanisms === "object" ? rawFam.law_mechanisms : {};
    const equations = Array.isArray(rawFam.equations)
      ? rawFam.equations
          .filter((eq) => eq && typeof eq === "object" && eq.latex)
          .map((eq) => ({
            label: String(eq.label || "Equation"),
            latex: String(eq.latex),
            explanation: String(eq.explanation || ""),
          }))
      : [];
    const quickstart =
      rawFam.quickstart && typeof rawFam.quickstart === "object"
        ? {
            preset: String(rawFam.quickstart.preset || id),
            snippet: Array.isArray(rawFam.quickstart.snippet)
              ? rawFam.quickstart.snippet.map(String)
              : typeof rawFam.quickstart.snippet === "string"
              ? rawFam.quickstart.snippet.split("\n")
              : [],
          }
        : null;
    const tags = Array.isArray(rawFam.tags) ? rawFam.tags.map(String).filter(Boolean) : [];

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
      pagesUrl: famPagesUrl,
      overview: rawFam.overview ? String(rawFam.overview) : "",
      law_mechanisms: lawMechanisms,
      equations,
      mermaid_diagram: rawFam.mermaid_diagram ? String(rawFam.mermaid_diagram) : "",
      quickstart,
      tags,
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

  function normalizeTaskDefinition(rawDef, fallbackId = "") {
    if (!rawDef || typeof rawDef !== "object") return null;
    const taskId = String(rawDef.task_id || rawDef.id || rawDef.test || fallbackId).trim();
    if (!taskId) return null;
    return {
      ...rawDef,
      task_id: taskId,
      title: String(rawDef.title || taskId.replace(/_/g, " ")),
      tier: rawDef.tier != null ? toInt(rawDef.tier, 0) : 0,
      role: normalizeRole(rawDef.role || "required"),
      views: Array.isArray(rawDef.views) ? rawDef.views.map(String) : [],
      summary: String(rawDef.summary || ""),
      re_entry_rule: rawDef.re_entry_rule || null,
      in_full_suite: Boolean(rawDef.in_full_suite),
      laws: Array.isArray(rawDef.laws) ? rawDef.laws.map(String).filter(Boolean) : [],
      edge_cases: Array.isArray(rawDef.edge_cases) ? rawDef.edge_cases.map(String).filter(Boolean) : [],
      intuition: String(rawDef.intuition || rawDef.summary || ""),
      failure_mode: String(rawDef.failure_mode || ""),
      protocol_notes: String(rawDef.protocol_notes || ""),
      prior_summary: String(rawDef.prior_summary || ""),
      model_summary: String(rawDef.model_summary || ""),
      steps: toNullableNum(rawDef.steps),
      thresholds: rawDef.thresholds && typeof rawDef.thresholds === "object" ? rawDef.thresholds : {},
      mermaid: rawDef.mermaid ? String(rawDef.mermaid) : "",
      svg_relpath: rawDef.svg_relpath ? String(rawDef.svg_relpath) : null,
      svg_inline: rawDef.svg_inline ? String(rawDef.svg_inline) : null,
      reproduction_cmd: String(rawDef.reproduction_cmd || `python -m experiments.forge probe unified-v11 --task ${taskId}`),
      pages_url: String(rawDef.pages_url || pagesUrl(`experiments/${taskId}/`)),
      task_config_path: rawDef.task_config_path ? String(rawDef.task_config_path) : null,
      adapter: rawDef.adapter ? String(rawDef.adapter) : "",
      sampling_law: rawDef.sampling_law ? String(rawDef.sampling_law) : "",
      scoring_weights: rawDef.scoring_weights ? String(rawDef.scoring_weights) : "",
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
      occam_timeouts: toNullableNum(rawRow.occam_timeouts),
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
    const role = normalizeRole(
      rawItem.role || rawItem.disposition || (rawItem.is_stretch ? "stretch" : rawItem.is_held ? "held" : "required")
    );
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
      gif_meta: rawItem.gif_meta && typeof rawItem.gif_meta === "object" ? rawItem.gif_meta : null,
      metrics: rawItem.metrics || rawItem.metrics_summary || null,
      raw: rawItem,
    };
  }

  function normalizeDataset(raw, hfApiSha) {
    const meta = state.meta;
    const rawMeta = raw.metadata || raw.provenance || raw;

    meta.datasetRepo = CONFIG.datasetRepo === "local" ? "local" : rawMeta.dataset_repo || CONFIG.datasetRepo;
    meta.pagesBaseUrl = String(raw.pages_base_url || rawMeta.pages_base_url || DEFAULT_PAGES_BASE_URL).replace(/\/+$/, "");
    meta.hfRevision =
      CONFIG.requestedRev !== "main"
        ? CONFIG.requestedRev
        : raw.hf_revision || rawMeta.hf_revision || hfApiSha || (isLocalMode() ? "local" : "main");
    const defaultMediaRev = isLocalMode() && meta.datasetRepo !== "local" ? "main" : meta.hfRevision;
    meta.mediaRevision = CONFIG.mediaRev || raw.media_hf_revision || rawMeta.media_hf_revision || defaultMediaRev;
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

    // Evolving Laws Registry (dynamic; safe fallback on older v1 payloads)
    const rawLawsMeta = raw.laws_meta && typeof raw.laws_meta === "object" ? raw.laws_meta : {};
    state.lawsMeta = {
      schema_version: toInt(rawLawsMeta.schema_version, 1),
      epistemic_status: String(rawLawsMeta.epistemic_status || "provisional_evolving"),
      epigraph: String(rawLawsMeta.epigraph || DEFAULT_EPISTEMIC_EPIGRAPH),
      epistemic_note: String(rawLawsMeta.epistemic_note || DEFAULT_EPISTEMIC_NOTE),
      doc_path: String(rawLawsMeta.doc_path || "docs/four-laws.md"),
      pages_url: String(rawLawsMeta.pages_url || pagesUrl("laws/")),
      source: String(rawLawsMeta.source || (Array.isArray(raw.laws) && raw.laws.length > 0 ? "data.laws" : "unavailable")),
    };
    const laws = Array.isArray(raw.laws) ? raw.laws.map(normalizeLaw).filter(Boolean) : [];
    const lawLookup = new Map();
    for (const law of laws) {
      lawLookup.set(law.id, { law, subLaw: null, parentId: law.id, label: `${law.short_label}: ${law.title}` });
      for (const sub of law.sub_laws) {
        lawLookup.set(sub.id, { law, subLaw: sub, parentId: law.id, label: `${sub.short_label}: ${sub.title}` });
      }
    }

    // Families list
    const rawFamilies = raw.families || raw.selected_recipes || raw.leaderboard || raw.selected_families || [];
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
        const norm = normalizeTaskDefinition(def);
        if (norm) taskDefinitions.set(norm.task_id, norm);
      }
    } else if (rawDefs && typeof rawDefs === "object") {
      for (const [tid, def] of Object.entries(rawDefs)) {
        const norm = normalizeTaskDefinition(def, tid);
        if (norm) taskDefinitions.set(norm.task_id, norm);
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
            receipts.push(
              normalizeReceipt({ family_id: f.id, family_label: f.label, task_id: taskId, ...item }, familyLookup, idx++)
            );
          }
        }
      }
    }

    // Optional separate media map
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
    state.laws = laws;
    state.lawLookup = lawLookup;
    state.families = families;
    state.receipts = receipts;
    state.energyBenchmarks = energyBenchmarks;
    state.taskDefinitions = taskDefinitions;
    state.fourLawsL4 =
      raw && typeof raw.four_laws_adherence_l4 === "object" && raw.four_laws_adherence_l4
        ? raw.four_laws_adherence_l4
        : null;
    state.fourLawsV2L4 =
      raw && typeof raw.four_laws_adherence_v2_l4 === "object" && raw.four_laws_adherence_v2_l4
        ? raw.four_laws_adherence_v2_l4
        : null;

    if (!state.selectedFormulationFamilyId || !families.some((f) => f.id === state.selectedFormulationFamilyId)) {
      const preferred =
        families.find((f) => f.id === "unified") ||
        families.find((f) => f.equations.length > 0 || f.mermaid_diagram) ||
        families[0] ||
        null;
      state.selectedFormulationFamilyId = preferred ? preferred.id : null;
    }
  }

  /* ---------- Law & Pareto Pure Helpers ---------- */

  function taskMatchesLaw(taskDef, lawFilterId) {
    if (!lawFilterId) return true;
    if (!taskDef || !Array.isArray(taskDef.laws) || taskDef.laws.length === 0) return false;
    if (taskDef.laws.includes(lawFilterId)) return true;
    const entry = state.lawLookup.get(lawFilterId);
    if (entry && !entry.subLaw && entry.law && Array.isArray(entry.law.sub_laws)) {
      const subIds = new Set(entry.law.sub_laws.map((s) => s.id));
      return taskDef.laws.some((tid) => subIds.has(tid));
    }
    return false;
  }

  function lawDisplayLabel(lawId) {
    const entry = state.lawLookup.get(lawId);
    if (entry) return entry.subLaw ? entry.subLaw.short_label : entry.law.short_label;
    return lawId;
  }

  function lawFullTitle(lawId) {
    const entry = state.lawLookup.get(lawId);
    if (entry) return entry.label;
    return lawId;
  }

  const COHORT_PALETTE = ["#4f46e5", "#0284c7", "#059669", "#d97706", "#db2777", "#7c3aed", "#0d9488"];

  function energyRowPriority(row) {
    const scope = String(row.evidence_scope || "");
    const isStudy = scope.includes("study") ? 1 : 0;
    const filled =
      (row.r256_joules_per_step != null ? 1 : 0) +
      (row.r256_ms_per_step != null ? 1 : 0) +
      (row.grid100_j_to_pass != null ? 1 : 0) +
      (row.ring16_j_to_pass != null ? 1 : 0) +
      (row.state_bytes_per_param != null ? 1 : 0);
    return isStudy * 100 + filled;
  }

  function candidateVersionNumber(candidateId) {
    const m = /-v(\d+)$/i.exec(String(candidateId || "").trim());
    return m ? Number(m[1]) : -1;
  }

  function buildParetoPoints(rows, families, xKey = "grid100_j_to_pass", yMode = "occam") {
    const famMap = new Map((families || []).map((f) => [f.id, f]));
    const cohortIndex = new Map();
    const candidateBest = new Map();

    // When both a dedicated study benchmark and receipt-aggregated row exist for the
    // exact same (candidate_id, hardware_gpu, contended), plot the richer study benchmark
    // once rather than double-plotting the same candidate on "All Cohorts".
    for (const row of rows || []) {
      const x = row[xKey];
      if (x == null || !Number.isFinite(x)) continue;

      let y = null;
      let yLabel = "";
      if (yMode === "full_suite") {
        const fam = row.family_id ? famMap.get(row.family_id) : null;
        if (fam && fam.full_total > 0) {
          y = (fam.full_pass / fam.full_total) * 100;
          yLabel = `${fam.full_pass}/${fam.full_total} (${y.toFixed(1)}%)`;
        }
      } else if (row.occam_pass != null && row.occam_executed != null && row.occam_executed > 0) {
        y = (row.occam_pass / row.occam_executed) * 100;
        const toNote = row.occam_timeouts > 0 ? `, ${row.occam_timeouts} timeout` : "";
        yLabel = `${row.occam_pass}/${row.occam_executed} (${y.toFixed(1)}%${toNote})`;
      }
      if (y == null || !Number.isFinite(y)) continue;

      const gpu = row.hardware_gpu || "GPU unrecorded";
      const contendLabel =
        row.contended === false ? "Uncontended" : row.contended === true ? "Contended" : "Contention unknown";
      const cohortKey = `${gpu} · ${contendLabel}`;
      const dedupKey = `${row.candidate_id}::${cohortKey}`;
      const prio = energyRowPriority(row);
      const prev = candidateBest.get(dedupKey);
      if (!prev || prio > prev.prio) {
        candidateBest.set(dedupKey, { row, x, y, yLabel, gpu, cohortKey, prio });
      }
    }

    const points = [];
    for (const item of candidateBest.values()) {
      const { row, x, y, yLabel, gpu, cohortKey } = item;
      if (!cohortIndex.has(cohortKey)) {
        cohortIndex.set(cohortKey, {
          key: cohortKey,
          gpu,
          contended: row.contended,
          color: COHORT_PALETTE[cohortIndex.size % COHORT_PALETTE.length],
          shape: row.contended === true ? "diamond" : "circle",
          count: 0,
        });
      }
      const cohort = cohortIndex.get(cohortKey);
      cohort.count += 1;

      points.push({
        key: row.key,
        candidate_id: row.candidate_id,
        family_id: row.family_id,
        optimizer_family: row.optimizer_family,
        hardware_gpu: gpu,
        contended: row.contended,
        cohortKey,
        color: cohort.color,
        shape: cohort.shape,
        x,
        y,
        yLabel,
        source_file: row.source_file,
        row,
        isFrontier: false,
        isLatestFamily: false,
      });
    }

    // Compute per-cohort Pareto frontier strictly for uncontended cohorts (minimize X, maximize Y)
    const frontiers = new Map();
    const byCohort = new Map();
    for (const pt of points) {
      if (!byCohort.has(pt.cohortKey)) byCohort.set(pt.cohortKey, []);
      byCohort.get(pt.cohortKey).push(pt);
    }
    for (const [cohortKey, group] of byCohort.entries()) {
      const cohort = cohortIndex.get(cohortKey);
      // Mark the latest versioned candidate per family_id in each cohort so major releases stay labeled.
      const latestByFam = new Map();
      for (const pt of group) {
        if (!pt.family_id) continue;
        const ver = candidateVersionNumber(pt.candidate_id);
        if (ver < 0) continue;
        const prev = latestByFam.get(pt.family_id);
        if (!prev || ver > prev.ver) latestByFam.set(pt.family_id, { ver, pt });
      }
      for (const { pt } of latestByFam.values()) {
        pt.isLatestFamily = true;
      }

      if (cohort.contended === true) continue;
      const ordered = [...group].sort((a, b) => a.x - b.x || b.y - a.y);
      const chain = [];
      let bestY = -Infinity;
      for (const pt of ordered) {
        if (pt.y > bestY + 1e-9) {
          pt.isFrontier = true;
          chain.push(pt);
          bestY = pt.y;
        }
      }
      if (chain.length > 0) frontiers.set(cohortKey, chain);
    }

    return {
      points,
      cohorts: [...cohortIndex.values()],
      frontiers,
    };
  }

  /* ---------- Rendering: Topbar, KPIs & Leaderboard ---------- */

  function setStatus(message, kind = "loading") {
    const banner = $("status-banner");
    const msg = $("status-message");
    if (!banner || !msg) return;
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
    const dsUrl =
      m.datasetRepo === "local"
        ? resolveDataUrl()
        : `https://huggingface.co/datasets/${m.datasetRepo}`;
    const portalHref = pagesUrl("");
    if ($("link-pages-portal")) $("link-pages-portal").href = portalHref;
    if ($("footer-portal-url")) $("footer-portal-url").href = portalHref;
    if ($("prov-dataset")) {
      $("prov-dataset").textContent = m.datasetRepo === "local" ? `local (${CONFIG.dataPath})` : m.datasetRepo;
      $("prov-dataset").href = dsUrl;
    }
    if ($("link-hf-dataset") && m.datasetRepo !== "local") $("link-hf-dataset").href = dsUrl;
    if ($("prov-hf-rev")) {
      $("prov-hf-rev").textContent = shortHash(m.hfRevision, 12);
      $("prov-hf-rev").title = m.hfRevision || "";
    }
    if ($("prov-git-commit")) {
      $("prov-git-commit").textContent = m.gitCommit ? shortHash(m.gitCommit, 10) : "develop";
      $("prov-git-commit").title = m.gitCommit || "";
    }
    if ($("prov-view")) {
      $("prov-view").textContent = m.viewRevision != null ? `${m.view} rev ${m.viewRevision}` : m.view;
    }
    if ($("prov-updated")) {
      $("prov-updated").textContent = m.generatedAt ? m.generatedAt.replace("T", " ").replace("Z", " UTC") : "—";
    }

    const dataUrl = isLocalMode() ? resolveDataUrl() : hfResolveUrl(CONFIG.dataPath, m.hfRevision);
    if ($("footer-data-url")) {
      $("footer-data-url").href = dataUrl;
      $("footer-data-url").textContent = `${CONFIG.dataPath} @ ${shortHash(m.hfRevision, 10)} ↗`;
    }

    if (m.gitCommit) {
      if ($("link-inventory")) $("link-inventory").href = githubBlobUrl("reports/forge/technique-inventory.md", m.gitCommit);
      if ($("link-problems")) $("link-problems").href = githubBlobUrl("reports/forge/problems.md", m.gitCommit);
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
    $("kpi-view-reqs-sub").textContent =
      m.distinctRequiredTests != null
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
    $("kpi-receipts-sub").textContent =
      m.backfillAttempted != null && m.backfillTotal != null
        ? `${distinctTasks} tasks · backfill ${m.backfillAttempted}/${m.backfillTotal} attempts`
        : `${distinctTasks} tasks · ${gifCount} with published GIFs`;

    const uncontendedEnergy = state.energyBenchmarks.filter((row) => row.contended !== true);
    const pool = uncontendedEnergy.length > 0 ? uncontendedEnergy : state.energyBenchmarks;
    const withGrid = pool.filter((row) => row.grid100_j_to_pass != null);
    const withRing = pool.filter((row) => row.ring16_j_to_pass != null);
    const withR256 = pool.filter((row) => row.r256_joules_per_step != null);
    if (withGrid.length > 0) {
      const bestGrid = [...withGrid].sort((a, b) => a.grid100_j_to_pass - b.grid100_j_to_pass)[0];
      const bestR256 =
        withR256.length > 0 ? [...withR256].sort((a, b) => a.r256_joules_per_step - b.r256_joules_per_step)[0] : null;
      $("kpi-energy").textContent = formatJoules(bestGrid.grid100_j_to_pass);
      $("kpi-energy-sub").textContent =
        `${bestGrid.candidate_id} (grid100)` +
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
    if (!table || !tbody) return;
    table.dataset.lbMode = state.lbMode;

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
        const staleBadge = f.stale
          ? `<span class="badge badge-stale" title="Recorded source measurement is stale; no current qualification">stale</span>`
          : "";
        const recipeLabel = f.selectedRecipe && f.selectedRecipe !== f.label ? `<code>${escapeHtml(f.selectedRecipe)}</code>` : "";
        const sourcePill = f.sourceHash
          ? `<code title="Source hash: ${escapeHtml(f.sourceHash)}">src:${escapeHtml(shortHash(f.sourceHash, 8))}</code>`
          : "";
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
    if (!card) return;
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
    if ($("fd-portal-link")) $("fd-portal-link").href = fam.pagesUrl;

    const heldSummary = fam.fullHeld
      ? `${fam.fullHeld.total}: ${fam.fullHeld.pass} PASS, ${fam.fullHeld.fail} FAIL, ${fam.fullHeld.not_measured} not measured, ${fam.fullHeld.other} other`
      : "—";
    const items = [
      ["Candidate ID", fam.candidateId || fam.selectedRecipe || "—"],
      ["Recipe SHA-256", fam.recipeHash || "—"],
      ["Source Digest", fam.sourceHash || "—"],
      ["Source Origin Commit", fam.sourceOriginCommit || "—"],
      ["Cohort Anchor", fam.cohortAnchor || "—"],
      [
        `Full-Suite (${fam.full_total || state.meta.fullSuiteTotal})`,
        `${fam.full_pass} PASS · ${fam.full_fail} FAIL · ${fam.full_not_measured} not measured · ${fam.full_other} other`,
      ],
      ["Held / Not Counted", heldSummary],
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

  /* ---------- Rendering: Evolving Laws & Families Matrix ---------- */

  function renderLawsSection() {
    const epigraphEl = $("laws-epigraph");
    const noteEl = $("laws-epistemic-note");
    const badgeEl = $("laws-status-badge");
    const docLink = $("link-laws-doc");
    const cardsGrid = $("laws-cards-grid");
    const matrixWrap = $("laws-matrix-wrap");
    if (!epigraphEl || !noteEl || !cardsGrid || !matrixWrap) return;

    const lm = state.lawsMeta;
    epigraphEl.textContent = `“${lm.epigraph}”`;
    noteEl.textContent = lm.epistemic_note;
    if (badgeEl) {
      const statusLabel = lm.epistemic_status === "provisional_evolving" ? "Provisional · Evolving Taxonomy" : lm.epistemic_status;
      badgeEl.textContent = state.laws.length > 0 ? `${statusLabel} (${state.laws.length} active)` : statusLabel;
    }
    if (docLink) {
      docLink.href = lm.pages_url || pagesUrl("laws/");
    }

    if (state.laws.length === 0) {
      cardsGrid.innerHTML = "";
      matrixWrap.innerHTML = `
        <div class="empty-cell">
          <strong>Evolving laws registry not embedded in this dataset snapshot.</strong>
          <div class="muted" style="margin-top:4px">
            This Space reads laws dynamically from <code>data.laws</code> (exported from <code>configs/forge/laws.json</code>) without hardcoding any fixed law taxonomy.
            Browse the family formulations, equations, and quickstarts below, or view <a href="${escapeHtml(githubBlobUrl(lm.doc_path))}" target="_blank" rel="noopener noreferrer">${escapeHtml(lm.doc_path)} ↗</a>.
          </div>
        </div>`;
      renderLawCellDrawer();
      renderFamilyFormulationSection();
      return;
    }

    // Dynamic Law Summary Cards
    const allDefs = [...state.taskDefinitions.values()];
    cardsGrid.innerHTML = state.laws
      .map((law) => {
        const taggedTasks = allDefs.filter((td) => taskMatchesLaw(td, law.id));
        const subChips = law.sub_laws
          .map(
            (sub) =>
              `<button type="button" class="law-sub-chip" data-filter-law-jump="${escapeHtml(sub.id)}" title="${escapeHtml(sub.title)}">${escapeHtml(sub.short_label)}</button>`
          )
          .join("");
        const docUrl = law.doc_anchor
          ? `${githubBlobUrl(lm.doc_path)}#${encodeURIComponent(law.doc_anchor)}`
          : githubBlobUrl(lm.doc_path);
        return `
          <article class="law-card">
            <div class="law-card-head">
              <h4 class="law-card-title">${escapeHtml(law.short_label)} · ${escapeHtml(law.title)}</h4>
              <a href="${escapeHtml(docUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-ghost btn-sm" title="Read specification">Spec ↗</a>
            </div>
            <p class="law-card-question">${escapeHtml(law.question)}</p>
            ${subChips ? `<div class="law-sub-chips">${subChips}</div>` : ""}
            <div class="law-card-foot">
              <span>${taggedTasks.length} tagged experiment${taggedTasks.length === 1 ? "" : "s"}</span>
              <button type="button" class="btn btn-ghost btn-sm" data-filter-law-jump="${escapeHtml(law.id)}">Filter Explorer →</button>
            </div>
          </article>
        `;
      })
      .join("");

    // Family x Law Coverage Heatmap
    const sortedFamilies = [...state.families].sort((a, b) => b.full_pass - a.full_pass || b.total_pass - a.total_pass);
    const headerCells = state.laws
      .map(
        (law) => `
        <th>
          <div>${escapeHtml(law.short_label)}</div>
          <div style="font-weight:500;text-transform:none;letter-spacing:0;font-size:0.71rem;color:var(--text-soft)">${escapeHtml(law.title)}</div>
        </th>`
      )
      .join("");

    const bodyRows = sortedFamilies
      .map((fam) => {
        const cells = state.laws
          .map((law) => {
            const entry = fam.law_mechanisms && fam.law_mechanisms[law.id];
            const meta = lawStatusMeta(entry ? entry.status : "unrecorded");
            const active =
              state.selectedLawCell &&
              state.selectedLawCell.familyId === fam.id &&
              state.selectedLawCell.lawId === law.id
                ? "active-cell"
                : "";
            const mechPreview = entry && entry.mechanism ? entry.mechanism : entry && entry.how_it_works ? entry.how_it_works : "Click to inspect";
            return `
              <td class="heatmap-td">
                <button
                  type="button"
                  class="heatmap-cell status-${escapeHtml(meta.key)} ${active}"
                  data-law-cell-family="${escapeHtml(fam.id)}"
                  data-law-cell-law="${escapeHtml(law.id)}"
                >
                  <span class="heatmap-cell-status">${meta.icon} ${escapeHtml(meta.label)}</span>
                  <span class="heatmap-cell-mech">${escapeHtml(mechPreview)}</span>
                </button>
              </td>`;
          })
          .join("");
        return `
          <tr>
            <td class="family-cell">
              <div class="family-name">${escapeHtml(fam.label)}</div>
              <div class="family-meta">
                <span class="badge badge-pass">${fam.full_pass}/${fam.full_total || state.meta.fullSuiteTotal}</span>
                ${fam.candidateId ? `<code>${escapeHtml(fam.candidateId)}</code>` : ""}
              </div>
            </td>
            ${cells}
          </tr>`;
      })
      .join("");

    matrixWrap.innerHTML = `
      <table class="data-table heatmap-table" id="laws-heatmap-table">
        <thead>
          <tr>
            <th>Family</th>
            ${headerCells}
          </tr>
        </thead>
        <tbody>${bodyRows}</tbody>
      </table>
    `;

    renderLawCellDrawer();
    renderFourLawsL4Section();
    renderFamilyFormulationSection();
  }

  function renderFourLawsL4Section() {
    const block = $("four-laws-l4-block");
    const wrap = $("four-laws-l4-wrap");
    if (!block || !wrap) return;
    const hasV2 = Boolean(state.fourLawsV2L4 && Array.isArray(state.fourLawsV2L4.rows) && state.fourLawsV2L4.rows.length > 0);
    const hasV1 = Boolean(state.fourLawsL4 && Array.isArray(state.fourLawsL4.rows) && state.fourLawsL4.rows.length > 0);
    if (!hasV2 && !hasV1) {
      block.classList.add("hidden");
      return;
    }
    const activeVer =
      state.fourLawsVersion === "v1" && hasV1
        ? "v1"
        : state.fourLawsVersion === "v2" && hasV2
          ? "v2"
          : hasV2
            ? "v2"
            : "v1";
    const fl = activeVer === "v2" ? state.fourLawsV2L4 : state.fourLawsL4;
    block.classList.remove("hidden");

    const toggleEl = $("four-laws-l4-toggle");
    if (toggleEl) {
      if (hasV2 && hasV1) {
        const v2Cnt = (state.fourLawsV2L4.probes || []).length || 10;
        const v1Cnt = (state.fourLawsL4.probes || []).length || 9;
        toggleEl.innerHTML = `
          <button type="button" class="seg-btn ${activeVer === "v2" ? "active" : ""}" data-fl-ver="v2">v2 Clean (${v2Cnt})</button>
          <button type="button" class="seg-btn ${activeVer === "v1" ? "active" : ""}" data-fl-ver="v1">v1 Historical (${v1Cnt})</button>
        `;
        toggleEl.querySelectorAll("[data-fl-ver]").forEach((btn) => {
          btn.addEventListener("click", () => {
            state.fourLawsVersion = btn.getAttribute("data-fl-ver") || "v2";
            renderFourLawsL4Section();
          });
        });
      } else {
        toggleEl.innerHTML = "";
      }
    }
    const descEl = $("four-laws-l4-desc");
    if (descEl) {
      const viewLabel = fl.view_id || (activeVer === "v2" ? "four-laws-adherence-v2" : "four-laws-adherence-v1");
      descEl.innerHTML = `Matched ${escapeHtml(fl.hardware_gpu || "NVIDIA L4")} hardware cohort (<code>${escapeHtml(viewLabel)}</code>, protocol seed 0, unchanged recipes; research diagnostic telemetry only — separate from ordinary selected-cohort qualification).`;
    }

    const probes = Array.isArray(fl.probes) ? fl.probes : [];
    const energyColLabel = `${probes.length || 9}-Probe NVML`;
    const probeHeaders = probes
      .map(
        (p) =>
          `<th class="num"><a href="${escapeHtml(p.pages_url || pagesUrl(`experiments/${p.task_id}/`))}" target="_blank" rel="noopener noreferrer">${escapeHtml(p.short_label || p.task_id)}</a></th>`
      )
      .join("");
    const rowsHtml = fl.rows
      .map((r) => {
        const famTag = r.family_id ? `<code>${escapeHtml(r.family_id)}</code>` : "—";
        if (r.blocked) {
          const bgTxt =
            r.bytegan_measured > 0 ? `${r.bytegan_pass}/${r.bytegan_measured}` : "BLOCKED";
          return `
            <tr>
              <td class="sticky-col"><strong><code>${escapeHtml(r.candidate_id)}</code></strong><div class="muted" style="font-size:0.74rem">${famTag}</div></td>
              <td class="num"><span class="badge badge-blocked">BLOCKED</span></td>
              <td class="num"><span class="badge badge-blocked">${escapeHtml(bgTxt)}</span></td>
              <td class="num">—</td>
              <td colspan="${Math.max(1, probes.length)}" class="muted">Statically blocked (${escapeHtml(r.blocked_reason || "contract")})</td>
            </tr>
          `;
        }
        const bgTxt = r.bytegan_measured > 0 ? `${r.bytegan_pass}/${r.bytegan_measured}` : "—";
        const bgCls =
          r.bytegan_measured > 0 && r.bytegan_pass === r.bytegan_measured
            ? "badge-pass"
            : "badge-notrun";
        const energyTxt =
          r.nine_probe_joules_total != null
            ? `${(Number(r.nine_probe_joules_total) / 1000).toFixed(2)} kJ`
            : "—";
        const probeTds = probes
          .map((p) => {
            const cell = r.probes && r.probes[p.task_id];
            if (!cell) return `<td class="num muted">—</td>`;
            const st = String(cell.status || "").toUpperCase();
            const mm =
              typeof cell.min_margin === "number" && Number.isFinite(cell.min_margin)
                ? `${cell.min_margin >= 0 ? "+" : ""}${cell.min_margin.toFixed(2)}`
                : "";
            if (st === "PASS") {
              const jtp =
                typeof cell.joules_to_pass === "number" && Number.isFinite(cell.joules_to_pass)
                  ? ` · ${(cell.joules_to_pass / 1000).toFixed(2)} kJ`
                  : "";
              return `<td class="num"><span class="badge badge-pass">PASS${escapeHtml(jtp)}</span></td>`;
            }
            if (st === "FAIL") {
              return `<td class="num"><span class="badge badge-fail">FAIL ${escapeHtml(mm)}</span></td>`;
            }
            return `<td class="num"><span class="badge badge-notrun">${escapeHtml(st)}</span></td>`;
          })
          .join("");
        return `
          <tr>
            <td class="sticky-col"><strong><code>${escapeHtml(r.candidate_id)}</code></strong><div class="muted" style="font-size:0.74rem">${famTag}</div></td>
            <td class="num"><strong>${r.probes_pass}/${r.probes_measured}</strong></td>
            <td class="num"><span class="badge ${bgCls}">${escapeHtml(bgTxt)}</span></td>
            <td class="num">${escapeHtml(energyTxt)}</td>
            ${probeTds}
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <table class="data-table" id="four-laws-l4-table">
        <thead>
          <tr>
            <th class="sticky-col">Candidate / Family</th>
            <th class="num">Probes PASS</th>
            <th class="num">ByteGAN (L4)</th>
            <th class="num">${escapeHtml(energyColLabel)}</th>
            ${probeHeaders}
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  }

  function renderLawCellDrawer() {
    const drawer = $("law-cell-drawer");
    if (!drawer) return;
    if (!state.selectedLawCell) {
      drawer.classList.add("hidden");
      return;
    }
    const { familyId, lawId } = state.selectedLawCell;
    const fam = state.families.find((f) => f.id === familyId);
    const law = state.laws.find((l) => l.id === lawId);
    if (!fam || !law) {
      drawer.classList.add("hidden");
      return;
    }
    drawer.classList.remove("hidden");
    const entry = fam.law_mechanisms && fam.law_mechanisms[law.id];
    const meta = lawStatusMeta(entry ? entry.status : "unrecorded");

    $("lcd-title").innerHTML = `${escapeHtml(fam.label)} × ${escapeHtml(law.short_label)}: ${escapeHtml(law.title)} <span class="badge ${meta.badgeCls}">${meta.icon} ${escapeHtml(meta.label)}</span>`;
    $("lcd-subtitle").textContent = law.question || `Candidate: ${fam.candidateId || fam.id}`;

    const boxes = [
      ["Mechanism", (entry && entry.mechanism) || "Not explicitly documented for this family snapshot."],
      ["How It Works (Plain English)", (entry && entry.how_it_works) || "See the family formulation card or report for detailed mechanics."],
      ["Known Limits & Honest Failure Modes", (entry && entry.known_limits) || "No additional known limits recorded in this snapshot."],
    ];
    $("lcd-body").innerHTML = boxes
      .map(
        ([label, val]) => `
        <div class="meta-box">
          <div class="meta-box-label">${escapeHtml(label)}</div>
          <div class="meta-box-value">${escapeHtml(val)}</div>
        </div>`
      )
      .join("");
  }

  function defaultQuickstartSnippet(fam) {
    if (fam.quickstart && fam.quickstart.snippet && fam.quickstart.snippet.length > 0) {
      return {
        preset: fam.quickstart.preset || fam.id,
        code: fam.quickstart.snippet.join("\n"),
        isDeclared: true,
      };
    }
    const presetGuess = fam.id === "unified" ? "unified_v11" : fam.id.replace(/-/g, "_");
    return {
      preset: presetGuess,
      code: [
        "import torch",
        "from particlegan import get_recipe",
        "",
        `recipe = get_recipe(${JSON.stringify(presetGuess)})`,
        "# See family report for exact candidate overrides and task host setup",
      ].join("\n"),
      isDeclared: false,
    };
  }

  function renderFamilyFormulationSection() {
    const switcher = $("family-switcher-pills");
    const card = $("family-formulation-card");
    if (!switcher || !card) return;

    if (state.families.length === 0) {
      switcher.innerHTML = "";
      card.innerHTML = `<div class="empty-cell">No family formulations available.</div>`;
      return;
    }

    switcher.innerHTML = state.families
      .map((f) => {
        const active = f.id === state.selectedFormulationFamilyId ? "active" : "";
        return `<button type="button" role="tab" aria-selected="${active ? "true" : "false"}" class="family-pill ${active}" data-formulation-family="${escapeHtml(f.id)}">${escapeHtml(f.label)}</button>`;
      })
      .join("");

    const fam = state.families.find((f) => f.id === state.selectedFormulationFamilyId) || state.families[0];
    if (!fam) return;

    const tagsHtml = fam.tags
      .map((tag) => `<span class="badge badge-muted">${escapeHtml(tag)}</span>`)
      .join("");

    const eqHtml =
      fam.equations.length > 0
        ? fam.equations
            .map(
              (eq) => `
              <div class="equation-item">
                <div class="equation-label">${escapeHtml(eq.label)}</div>
                <div class="equation-math">${renderLatexHtml(eq.latex, true)}</div>
                ${eq.explanation ? `<p class="equation-expl">${renderInlineMathHtml(eq.explanation)}</p>` : ""}
              </div>`
            )
            .join("")
        : `<p class="muted" style="margin:0;font-size:0.82rem">No equations embedded for this family snapshot. See the full <a href="${escapeHtml(githubBlobUrl(fam.pagePath))}" target="_blank" rel="noopener noreferrer">Family Report ↗</a>.</p>`;

    const mermaidSrc =
      fam.mermaid_diagram ||
      [
        "flowchart LR",
        `  P["Prior P"] --> G["Generator G (${fam.label.replace(/["`<>]/g, "")})"]`,
        `  G --> D["Critic D + Regularizer"]`,
        `  D --> U["Step G, D, P (${(fam.candidateId || fam.id).replace(/["`<>]/g, "")})"]`,
      ].join("\n");

    const qs = defaultQuickstartSnippet(fam);

    card.innerHTML = `
      <div class="formulation-head">
        <div>
          <h4>${escapeHtml(fam.label)} ${fam.candidateId ? `<code>${escapeHtml(fam.candidateId)}</code>` : ""}</h4>
          <div class="muted" style="font-size:0.8rem">
            Full-suite: <strong>${fam.full_pass}/${fam.full_total || state.meta.fullSuiteTotal} PASS</strong>
            · View Reqs: <strong>${fam.total_pass}/${fam.total_req}</strong>
            ${fam.stale ? ` · <span class="badge badge-stale">stale</span>` : ""}
          </div>
        </div>
        <div class="detail-drawer-actions">
          <a class="btn btn-ghost btn-sm" href="${escapeHtml(fam.pagesUrl)}" target="_blank" rel="noopener noreferrer">Portal Page ↗</a>
          <a class="btn btn-ghost btn-sm" href="${escapeHtml(githubBlobUrl(fam.pagePath))}" target="_blank" rel="noopener noreferrer">GitHub Report ↗</a>
        </div>
      </div>

      ${fam.overview ? `<p class="formulation-overview">${escapeHtml(fam.overview)}</p>` : ""}
      ${tagsHtml ? `<div class="tag-row">${tagsHtml}</div>` : ""}

      <div class="formulation-grid">
        <div class="formulation-pane formulation-pane-wide">
          <div class="formulation-pane-head">
            <h5>Update Loop Diagram (Mermaid)</h5>
            <span class="badge badge-muted">${fam.mermaid_diagram ? "Declared" : "Summary flow"}</span>
          </div>
          <div class="mermaid-host" data-mermaid-source="${escapeHtml(mermaidSrc)}">
            <pre class="mermaid-fallback">${escapeHtml(mermaidSrc)}</pre>
          </div>
        </div>

        <div class="formulation-pane">
          <div class="formulation-pane-head">
            <h5>Mathematical Formulation (KaTeX)</h5>
            <span class="badge badge-muted">${fam.equations.length} eq</span>
          </div>
          ${eqHtml}
        </div>

        <div class="formulation-pane">
          <div class="formulation-pane-head">
            <h5>PyTorch Quickstart</h5>
            <button type="button" class="btn btn-ghost btn-sm" data-copy-snippet="1">Copy Snippet</button>
          </div>
          <div class="code-block-wrap">
            <pre class="code-pre"><code id="family-quickstart-code">${escapeHtml(qs.code)}</code></pre>
          </div>
        </div>
      </div>
    `;

    const copyBtn = card.querySelector("[data-copy-snippet]");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => copyTextWithFeedback(qs.code, copyBtn));
    }
    renderMermaidBlocks(card);
  }

  /* ---------- Rendering: Energy & Pareto Scatter Plot ---------- */

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

  function formatParetoAxisValue(xKey, value) {
    if (xKey === "r256_ms_per_step") return `${formatNum(value, 1)} ms`;
    if (xKey === "r256_joules_per_step") return `${formatNum(value, 2)} J`;
    return formatJoules(value);
  }

  function paretoAxisTitle(xKey) {
    if (xKey === "r256_ms_per_step") return "DCGAN r256 Step Time (ms/step · lower is faster)";
    if (xKey === "r256_joules_per_step") return "DCGAN r256 Step Energy (J/step · lower is leaner)";
    if (xKey === "ring16_j_to_pass") return "ring16 J-to-PASS (Joules to first PASS · lower is better)";
    return "grid100 J-to-PASS (Joules to first PASS · lower is better)";
  }

  function renderParetoChart(filteredRows) {
    const wrap = $("pareto-svg-wrap");
    const legendEl = $("pareto-legend");
    const subEl = $("pareto-subtitle");
    const tooltip = $("pareto-tooltip");
    if (!wrap || !legendEl) return;

    document.querySelectorAll("[data-pareto-x]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.paretoX === state.paretoX);
    });
    document.querySelectorAll("[data-pareto-y]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.paretoY === state.paretoY);
    });

    const { points, cohorts, frontiers } = buildParetoPoints(filteredRows, state.families, state.paretoX, state.paretoY);

    if (subEl) {
      subEl.textContent =
        state.paretoY === "full_suite"
          ? "Y-axis matches each benchmark row's family full-suite pass rate (separate screening cohort). GPU & contention cohorts remain separated."
          : "Y-axis shows each row's committed Occam Suite Pass % vs. hardware NVML cost. Each GPU & contention cohort is colored and frontier-traced separately.";
    }

    if (points.length === 0) {
      wrap.innerHTML = `<div class="empty-cell">No benchmark rows have both <code>${escapeHtml(state.paretoX)}</code> and ${state.paretoY === "occam" ? "Occam pass rate" : "family full-suite pass rate"} under the current filter. Try switching the X/Y toggle or cohort filter.</div>`;
      legendEl.innerHTML = "";
      if (tooltip) tooltip.classList.add("hidden");
      return;
    }

    const W = 840;
    const H = 345;
    const pad = { top: 38, right: 32, bottom: 52, left: 66 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    let xMin = Math.min(...xs);
    let xMax = Math.max(...xs);
    const xSpan = xMax - xMin;
    xMin = Math.max(0, xMin - (xSpan > 0 ? xSpan * 0.1 : Math.max(1, xMin * 0.15)));
    xMax = xMax + (xSpan > 0 ? xSpan * 0.12 : Math.max(1, xMax * 0.15));

    const yMin = Math.max(0, Math.floor((Math.min(...ys) - 8) / 10) * 10);
    const yTickMax = Math.min(100, Math.max(100, Math.ceil(Math.max(...ys) / 10) * 10));
    const yPlotMax = yTickMax + 4;

    const sx = (v) => pad.left + ((v - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
    const sy = (v) => pad.top + (1 - (v - yMin) / Math.max(1e-9, yPlotMax - yMin)) * plotH;

    // Gridlines
    const yTicks = 5;
    const xTicks = 5;
    let gridSvg = "";
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + ((yTickMax - yMin) * i) / yTicks;
      const py = sy(val);
      gridSvg += `<line x1="${pad.left}" y1="${py.toFixed(1)}" x2="${W - pad.right}" y2="${py.toFixed(1)}" stroke="var(--border)" stroke-dasharray="3 3" />`;
      gridSvg += `<text x="${pad.left - 10}" y="${(py + 4).toFixed(1)}" text-anchor="end" fill="var(--text-muted)" font-size="11">${val.toFixed(0)}%</text>`;
    }
    for (let i = 0; i <= xTicks; i++) {
      const val = xMin + ((xMax - xMin) * i) / xTicks;
      const px = sx(val);
      gridSvg += `<line x1="${px.toFixed(1)}" y1="${pad.top}" x2="${px.toFixed(1)}" y2="${H - pad.bottom}" stroke="var(--border)" stroke-dasharray="3 3" />`;
      gridSvg += `<text x="${px.toFixed(1)}" y="${H - pad.bottom + 18}" text-anchor="middle" fill="var(--text-muted)" font-size="11">${escapeHtml(formatParetoAxisValue(state.paretoX, val))}</text>`;
    }

    // Subtle optimal-direction hint cleanly placed in the top margin above plotted points
    const hintSvg = `
      <rect x="${pad.left}" y="8" width="224" height="22" rx="11" fill="var(--accent-soft)" />
      <text x="${pad.left + 12}" y="23" fill="var(--accent)" font-size="11" font-weight="600">← Better: lower cost, higher pass ↑</text>
    `;

    // Frontier polylines (strictly within each uncontended cohort)
    let frontierSvg = "";
    for (const [, chain] of frontiers.entries()) {
      if (chain.length < 2) continue;
      const d = chain
        .map((pt, idx) => `${idx === 0 ? "M" : "L"}${sx(pt.x).toFixed(1)},${sy(pt.y).toFixed(1)}`)
        .join(" ");
      frontierSvg += `<path d="${d}" fill="none" stroke="${chain[0].color}" stroke-width="2.2" stroke-dasharray="6 4" opacity="0.85" />`;
    }

    // Points + collision-avoided labels for frontier, latest-per-family, and selected points
    const placedLabels = [];
    const labelPlacements = new Map();
    const labelCandidates = points
      .map((pt, idx) => ({ pt, idx, cx: sx(pt.x), cy: sy(pt.y), selected: state.selectedEnergyKey === pt.key }))
      .filter(({ pt, selected }) => selected || pt.isFrontier || pt.isLatestFamily || points.length <= 10)
      .sort((a, b) => {
        const pa = (a.selected ? 4 : 0) + (a.pt.isFrontier ? 2 : 0) + (a.pt.isLatestFamily ? 1 : 0);
        const pb = (b.selected ? 4 : 0) + (b.pt.isFrontier ? 2 : 0) + (b.pt.isLatestFamily ? 1 : 0);
        return pb - pa || a.cx - b.cx;
      });
    const labelOffsets = [
      [10, 4, "start"],
      [-10, 4, "end"],
      [10, -10, "start"],
      [-10, -10, "end"],
      [10, 16, "start"],
      [-10, 16, "end"],
      [0, -13, "middle"],
      [0, 19, "middle"],
    ];
    for (const item of labelCandidates) {
      const estW = Math.max(42, item.pt.candidate_id.length * 6.3 + 6);
      const estH = 13;
      let chosen = null;
      for (const [dx, dy, anchor] of labelOffsets) {
        const lx = item.cx + dx;
        const ly = item.cy + dy;
        const x0 = anchor === "start" ? lx : anchor === "end" ? lx - estW : lx - estW / 2;
        const x1 = x0 + estW;
        const y0 = ly - estH + 2;
        const y1 = ly + 2;
        if (x0 < pad.left + 2 || x1 > W - pad.right - 2 || y0 < pad.top - 4 || y1 > H - pad.bottom - 2) continue;
        const overlaps = placedLabels.some((b) => !(x1 + 3 < b.x0 || x0 - 3 > b.x1 || y1 + 2 < b.y0 || y0 - 2 > b.y1));
        if (!overlaps) {
          chosen = { lx, ly, anchor, box: { x0, x1, y0, y1 } };
          break;
        }
      }
      if (!chosen && (item.selected || item.pt.isFrontier || item.pt.isLatestFamily)) {
        const anchorRight = item.cx > W - pad.right - 135;
        const lx = anchorRight ? item.cx - 10 : item.cx + 10;
        const anchor = anchorRight ? "end" : "start";
        const ly = Math.min(H - pad.bottom - 6, Math.max(pad.top + 10, item.cy + 4));
        const x0 = anchor === "start" ? lx : lx - estW;
        chosen = { lx, ly, anchor, box: { x0, x1: x0 + estW, y0: ly - estH + 2, y1: ly + 2 } };
      }
      if (chosen) {
        placedLabels.push(chosen.box);
        labelPlacements.set(item.idx, chosen);
      }
    }

    const pointsSvg = points
      .map((pt, idx) => {
        const cx = sx(pt.x);
        const cy = sy(pt.y);
        const selected = state.selectedEnergyKey === pt.key;
        const r = selected ? 8 : pt.isFrontier ? 6.5 : pt.isLatestFamily ? 6.0 : 5.2;
        const stroke = selected ? "var(--text)" : "var(--surface)";
        const strokeW = selected ? 2.5 : 1.6;
        let marker = "";
        if (pt.shape === "diamond") {
          const s = r + 1.2;
          marker = `<polygon points="${cx.toFixed(1)},${(cy - s).toFixed(1)} ${(cx + s).toFixed(1)},${cy.toFixed(1)} ${cx.toFixed(1)},${(cy + s).toFixed(1)} ${(cx - s).toFixed(1)},${cy.toFixed(1)}" fill="${pt.color}" fill-opacity="0.72" stroke="${stroke}" stroke-width="${strokeW}" />`;
        } else {
          marker = `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}" fill="${pt.color}" stroke="${stroke}" stroke-width="${strokeW}" />`;
        }
        const halo = pt.isFrontier
          ? `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r + 3.5}" fill="none" stroke="${pt.color}" stroke-width="1.2" opacity="0.45" />`
          : "";
        const placement = labelPlacements.get(idx);
        const label = placement
          ? `<text x="${placement.lx.toFixed(1)}" y="${placement.ly.toFixed(1)}" text-anchor="${placement.anchor}" fill="var(--text)" font-size="10.5" font-weight="${pt.isFrontier || pt.isLatestFamily ? "700" : "500"}">${escapeHtml(pt.candidate_id)}</text>`
          : "";
        return `<g class="pareto-point" data-pareto-idx="${idx}" data-energy-key="${escapeHtml(pt.key)}" tabindex="0" role="button" aria-label="${escapeHtml(`${pt.candidate_id}: ${formatParetoAxisValue(state.paretoX, pt.x)}, ${pt.yLabel}`)}">${halo}${marker}${label}</g>`;
      })
      .join("");

    const yAxisTitle = state.paretoY === "full_suite" ? "Family Full-Suite Pass (%)" : "Occam Suite Pass (%)";

    wrap.innerHTML = `
      <svg id="pareto-svg" class="pareto-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Pareto efficiency scatter plot">
        <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${H - pad.bottom}" stroke="var(--border-strong)" />
        <line x1="${pad.left}" y1="${H - pad.bottom}" x2="${W - pad.right}" y2="${H - pad.bottom}" stroke="var(--border-strong)" />
        ${gridSvg}
        ${hintSvg}
        ${frontierSvg}
        ${pointsSvg}
        <text x="${(pad.left + W - pad.right) / 2}" y="${H - 12}" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-weight="600">${escapeHtml(paretoAxisTitle(state.paretoX))}</text>
        <text x="18" y="${(pad.top + H - pad.bottom) / 2}" text-anchor="middle" fill="var(--text-muted)" font-size="12" font-weight="600" transform="rotate(-90 18 ${(pad.top + H - pad.bottom) / 2})">${escapeHtml(yAxisTitle)}</text>
      </svg>
    `;

    legendEl.innerHTML =
      cohorts
        .map(
          (c) => `
          <span class="legend-item">
            <span class="legend-swatch ${c.shape === "diamond" ? "diamond" : ""}" style="background:${c.color}"></span>
            <span>${escapeHtml(c.key)} (${c.count})</span>
          </span>`
        )
        .join("") +
      `<span class="legend-item"><span style="border-top:2px dashed var(--text-muted);width:18px;display:inline-block"></span><span>Per-cohort Pareto frontier (uncontended only)</span></span>`;

    // Interactive hover + click handlers
    const svg = $("pareto-svg");
    if (svg && tooltip) {
      const stageRect = () => wrap.parentElement.getBoundingClientRect();
      svg.querySelectorAll(".pareto-point").forEach((node) => {
        const pt = points[Number(node.dataset.paretoIdx)];
        if (!pt) return;
        const showTip = (ev) => {
          const rect = stageRect();
          tooltip.innerHTML = `
            <strong>${escapeHtml(pt.candidate_id)} ${pt.isFrontier ? '<span class="badge badge-pass">Pareto</span>' : ""}</strong>
            <div>${escapeHtml(pt.optimizer_family || pt.family_id || "—")}</div>
            <div>X: <strong>${escapeHtml(formatParetoAxisValue(state.paretoX, pt.x))}</strong> · Y: <strong>${escapeHtml(pt.yLabel)}</strong></div>
            <div class="muted" style="margin-top:2px">${escapeHtml(pt.cohortKey)}</div>
            <div class="muted">Click point to inspect telemetry</div>
          `;
          tooltip.classList.remove("hidden");
          const left = Math.min(Math.max(12, ev.clientX - rect.left + 14), Math.max(12, rect.width - 295));
          const top = Math.max(10, ev.clientY - rect.top - 12);
          tooltip.style.left = `${left}px`;
          tooltip.style.top = `${top}px`;
        };
        node.addEventListener("mouseenter", showTip);
        node.addEventListener("mousemove", showTip);
        node.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
        const selectPoint = () => {
          state.selectedEnergyKey = pt.key;
          renderEnergyTable();
          renderEnergyDetail();
        };
        node.addEventListener("click", selectPoint);
        node.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            selectPoint();
          }
        });
      });
    }
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
    renderParetoChart(filtered);

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
      [
        "GPU Cohort",
        `${row.hardware_gpu || "—"} · ${row.contended === false ? "uncontended" : row.contended === true ? "contended" : "contention unknown"}`,
      ],
      [
        "r256 Step Cost",
        `${row.r256_ms_per_step != null ? `${formatNum(row.r256_ms_per_step, 2)} ms/step` : "—"} · ${row.r256_joules_per_step != null ? `${formatNum(row.r256_joules_per_step, 2)} J/step` : "—"} · ${row.r256_watts != null ? `${formatNum(row.r256_watts, 1)} W` : "—"}`,
      ],
      ["Optimizer State Footprint", row.state_bytes_per_param != null ? `${formatStateBpp(row.state_bytes_per_param)} B/param` : "—"],
      ["ring16 J-to-PASS", row.ring16_j_to_pass != null ? `${formatJoules(row.ring16_j_to_pass)} (first pass @ step ${row.ring16_first_pass_step ?? "—"})` : "—"],
      ["grid100 J-to-PASS (First)", row.grid100_j_to_pass != null ? `${formatJoules(row.grid100_j_to_pass)} (@ step ${row.grid100_first_pass_step ?? "—"})` : "—"],
      [
        "grid100 J-to-Sustained PASS",
        row.grid100_j_to_sustained_pass != null ? `${formatJoules(row.grid100_j_to_sustained_pass)} (@ step ${row.grid100_sustained_pass_step ?? "—"})` : "—",
      ],
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

  /* ---------- Rendering: Task & Receipt Explorer ---------- */

  function buildLawFilterOptionsHtml() {
    const opts = [`<option value="">All laws</option>`];
    const seen = new Set();
    for (const law of state.laws) {
      seen.add(law.id);
      opts.push(`<option value="${escapeHtml(law.id)}">${escapeHtml(law.short_label)} · ${escapeHtml(law.title)}</option>`);
      for (const sub of law.sub_laws) {
        seen.add(sub.id);
        opts.push(`<option value="${escapeHtml(sub.id)}">↳ ${escapeHtml(sub.short_label)} · ${escapeHtml(sub.title)}</option>`);
      }
    }
    // Include any law tags present in taskDefinitions even if registry isn't loaded
    const extra = new Set();
    for (const td of state.taskDefinitions.values()) {
      for (const lid of td.laws || []) {
        if (!seen.has(lid)) extra.add(lid);
      }
    }
    for (const lid of [...extra].sort()) {
      opts.push(`<option value="${escapeHtml(lid)}">${escapeHtml(lid)}</option>`);
    }
    return opts.join("");
  }

  function populateFilterOptions() {
    const famSelect = $("filter-family");
    if (famSelect) {
      const prevFam = state.expFilters.family;
      famSelect.innerHTML =
        `<option value="">All families (${state.families.length})</option>` +
        state.families.map((f) => `<option value="${escapeHtml(f.id)}">${escapeHtml(f.label)}</option>`).join("");
      famSelect.value = prevFam;
    }

    const lawHtml = buildLawFilterOptionsHtml();
    const lawSelect = $("filter-law");
    if (lawSelect) {
      lawSelect.innerHTML = lawHtml;
      lawSelect.value = state.expFilters.law;
    }
    const gifLawSelect = $("gif-law-filter");
    if (gifLawSelect) {
      gifLawSelect.innerHTML = lawHtml;
      gifLawSelect.value = state.gifLawFilter;
    }

    const edgeSelect = $("filter-edge-case");
    if (edgeSelect) {
      const edges = new Set();
      for (const td of state.taskDefinitions.values()) {
        for (const ec of td.edge_cases || []) edges.add(ec);
      }
      const sortedEdges = [...edges].sort();
      edgeSelect.innerHTML =
        `<option value="">All edge cases${sortedEdges.length ? ` (${sortedEdges.length})` : ""}</option>` +
        sortedEdges.map((ec) => `<option value="${escapeHtml(ec)}">${escapeHtml(ec)}</option>`).join("");
      edgeSelect.value = state.expFilters.edgeCase;
    }
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
      const def = state.taskDefinitions.get(r.task_id);
      if (f.law && !taskMatchesLaw(def, f.law)) return false;
      if (f.edgeCase && (!def || !Array.isArray(def.edge_cases) || !def.edge_cases.includes(f.edgeCase))) return false;
      if (q) {
        const extra = def ? `${def.title} ${(def.laws || []).join(" ")} ${(def.edge_cases || []).join(" ")} ${def.intuition || ""}` : "";
        const hay = `${r.task_id} ${r.family_id} ${r.family_label} ${r.candidate_id} ${r.status} ${r.role} ${r.reason} ${r.recipe_hash} ${r.source_hash} ${r.attempt_id} ${extra}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function renderExplorer() {
    const table = $("explorer-table");
    const tbody = $("explorer-tbody");
    if (!table || !tbody) return;

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
        const def = state.taskDefinitions.get(r.task_id);
        const lawTags =
          def && def.laws.length > 0
            ? `<div class="family-meta">${def.laws
                .slice(0, 3)
                .map((lid) => `<span class="badge badge-law" data-filter-law-inline="${escapeHtml(lid)}">${escapeHtml(lawDisplayLabel(lid))}</span>`)
                .join("")}</div>`
            : "";
        const provPills = [
          r.recipe_hash ? `rec:${shortHash(r.recipe_hash, 8)}` : "",
          r.source_hash ? `src:${shortHash(r.source_hash, 8)}` : "",
        ].filter(Boolean);
        const gifBtn = r.gif_path
          ? `<button type="button" class="btn btn-sm btn-primary" data-gif-jump="${escapeHtml(r.task_id)}" data-gif-fam="${escapeHtml(r.family_id)}">GIF</button>`
          : `<button type="button" class="btn btn-sm btn-ghost" data-gif-jump="${escapeHtml(r.task_id)}" data-gif-fam="${escapeHtml(r.family_id)}" title="Open in GIF Inspector">Inspect</button>`;

        return `
          <tr class="clickable ${selCls}" data-receipt-key="${escapeHtml(r.key)}">
            <td>
              <button type="button" class="task-link-btn" data-open-experiment="${escapeHtml(r.task_id)}" title="Open Experiment Deep-Dive">
                <code>${escapeHtml(r.task_id)}</code>
                ${def && def.title && def.title !== r.task_id ? `<span class="task-link-title">${escapeHtml(def.title)}</span>` : ""}
              </button>
              ${lawTags}
            </td>
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
    if (!card) return;
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

    let html =
      `<div class="detail-meta-grid">` +
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

  /* ---------- Rendering: GIF Inspector ---------- */

  const FLAGSHIP_VISUAL_TASKS = [
    "ring16_acquisition",
    "grid100",
    "rotated100",
    "staggered100",
    "vector_spiral",
    "img_bars4",
    "img_stripes2",
    "img_blobs4",
    "two_pole",
    "gaussian1d_smoke",
    "ae_gan_hold",
    "vector_two_broad",
  ];

  function rankFamiliesByPerformance(families) {
    return [...(families || [])].sort((a, b) => {
      const fpDiff = toInt(b.full_pass, 0) - toInt(a.full_pass, 0);
      if (fpDiff !== 0) return fpDiff;
      const tpDiff = toInt(b.total_pass, 0) - toInt(a.total_pass, 0);
      if (tpDiff !== 0) return tpDiff;
      const t1Diff = toInt(b.tier1_pass, 0) - toInt(a.tier1_pass, 0);
      if (t1Diff !== 0) return t1Diff;
      const failDiff = toInt(a.full_fail, 0) - toInt(b.full_fail, 0);
      if (failDiff !== 0) return failDiff;
      const staleDiff = (a.stale ? 1 : 0) - (b.stale ? 1 : 0);
      if (staleDiff !== 0) return staleDiff;
      return String(a.label || a.id).localeCompare(String(b.label || b.id));
    });
  }

  function buildFamilyRankMap(families = state.families) {
    const ranked = rankFamiliesByPerformance(families);
    const map = new Map();
    ranked.forEach((f, idx) => map.set(f.id, idx + 1));
    return { ranked, rankMap: map };
  }

  function pickDefaultGifFamilies(taskId, rankedFamilies, receipts, count = 2) {
    const ranked = Array.isArray(rankedFamilies) ? rankedFamilies : [];
    const byFam = new Map();
    for (const r of receipts || []) {
      if (r.task_id === taskId) byFam.set(r.family_id, r);
    }
    const withGif = ranked.filter((f) => Boolean(byFam.get(f.id)?.gif_path));
    const withoutGif = ranked.filter((f) => !byFam.get(f.id)?.gif_path);
    return [...withGif, ...withoutGif].slice(0, count).map((f) => f.id);
  }

  function pickDefaultGifTask(tasks, rankedFamilies, receipts) {
    if (!tasks || tasks.length === 0) return "";
    const top4Ids = new Set((rankedFamilies || []).slice(0, 4).map((f) => f.id));
    const top2Ids = new Set((rankedFamilies || []).slice(0, 2).map((f) => f.id));
    const taskStats = new Map();
    for (const t of tasks) {
      taskStats.set(t.taskId, { taskId: t.taskId, top2: 0, top4: 0, gifCount: t.gifCount || 0 });
    }
    for (const r of receipts || []) {
      if (!r.gif_path) continue;
      const st = taskStats.get(r.task_id);
      if (!st) continue;
      if (top2Ids.has(r.family_id)) st.top2 += 1;
      if (top4Ids.has(r.family_id)) st.top4 += 1;
    }

    // Prefer flagship visual benchmarks where the top-2 best-performing families have GIFs
    for (const flagship of FLAGSHIP_VISUAL_TASKS) {
      const st = taskStats.get(flagship);
      if (st && st.top2 >= Math.min(2, top2Ids.size) && st.gifCount > 0) {
        return flagship;
      }
    }
    for (const flagship of FLAGSHIP_VISUAL_TASKS) {
      const st = taskStats.get(flagship);
      if (st && st.top4 > 0) {
        return flagship;
      }
    }

    const best = [...taskStats.values()].sort((a, b) => {
      if (b.top2 !== a.top2) return b.top2 - a.top2;
      if (b.top4 !== a.top4) return b.top4 - a.top4;
      if (b.gifCount !== a.gifCount) return b.gifCount - a.gifCount;
      return a.taskId.localeCompare(b.taskId);
    })[0];
    return best ? best.taskId : tasks[0].taskId;
  }

  function pickContrastGifPair(taskId, rankedFamilies, receipts) {
    const byFam = new Map();
    for (const r of receipts || []) {
      if (r.task_id === taskId) byFam.set(r.family_id, r);
    }
    const passWithGif = rankedFamilies.find((f) => {
      const r = byFam.get(f.id);
      return r && r.gif_path && r.status === "PASS";
    });
    const failWithGif = rankedFamilies.find((f) => {
      const r = byFam.get(f.id);
      return r && r.gif_path && r.status === "FAIL";
    });
    if (passWithGif && failWithGif) {
      return [passWithGif.id, failWithGif.id];
    }
    const passAny = rankedFamilies.find((f) => byFam.get(f.id)?.status === "PASS");
    const failAny = rankedFamilies.find((f) => byFam.get(f.id)?.status === "FAIL");
    if (passAny && failAny) {
      return [passAny.id, failAny.id];
    }
    return pickDefaultGifFamilies(taskId, rankedFamilies, receipts, 2);
  }

  function gifDisplayUrl(gifPath) {
    const base = hfResolveUrl(gifPath);
    if (!base) return null;
    if (!state.gifSyncNonce) return base;
    const sep = base.includes("?") ? "&" : "?";
    return `${base}${sep}replay=${state.gifSyncNonce}`;
  }

  function formatGifProvenance(receipt) {
    const gm = receipt && receipt.gif_meta;
    if (!gm) return null;
    const parts = [];
    if (gm.cohort_match === "selected_cohort") {
      parts.push("Selected cohort");
    } else if (gm.cohort_match === "colab_replay_media") {
      parts.push(`Colab${gm.gpu_name ? ` ${gm.gpu_name.replace(/^NVIDIA\s+/i, "")}` : ""} seed-0 replay`);
    } else if (gm.cohort_match) {
      parts.push(String(gm.cohort_match));
    }
    parts.push(gm.has_visual_samples ? "Samples + curves" : "Metric curves");
    if (gm.grade_matches_selected === false && gm.recorded_grade) {
      parts.push(`GIF run: ${gm.recorded_grade}`);
    }
    return parts.join(" · ");
  }

  function compactGifMetricsHtml(receipt) {
    if (!receipt) return "";
    const pills = [];
    if (receipt.wall_seconds != null) {
      pills.push(`<span class="gif-metric-pill">wall ${receipt.wall_seconds.toFixed(1)}s</span>`);
    }
    if (receipt.energy && receipt.energy.joules_to_first_pass != null) {
      pills.push(`<span class="gif-metric-pill">J-to-PASS ${escapeHtml(formatJoules(receipt.energy.joules_to_first_pass))}</span>`);
    }
    if (receipt.metrics && typeof receipt.metrics === "object") {
      const entries = Object.entries(receipt.metrics)
        .filter(([k, v]) => !k.startsWith("_") && (typeof v === "number" || typeof v === "boolean"))
        .slice(0, 3);
      for (const [k, v] of entries) {
        const val = typeof v === "number" ? (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : v.toFixed(3)) : String(v);
        pills.push(`<span class="gif-metric-pill"><code>${escapeHtml(k)}</code>=${escapeHtml(val)}</span>`);
      }
    }
    return pills.length > 0 ? `<div class="gif-metrics-row">${pills.join("")}</div>` : "";
  }

  function populateGifSelectors({ snapFamiliesToBest = false } = {}) {
    const { ranked } = buildFamilyRankMap(state.families);
    const taskMap = new Map();
    for (const r of state.receipts) {
      if (!taskMap.has(r.task_id)) {
        taskMap.set(r.task_id, { taskId: r.task_id, tier: r.tier, role: r.role, gifCount: 0, passCount: 0 });
      }
      const entry = taskMap.get(r.task_id);
      if (r.gif_path) entry.gifCount += 1;
      if (r.status === "PASS") entry.passCount += 1;
    }

    const allTasks = [...taskMap.values()];
    const quickWrap = $("gif-quick-tasks");
    if (quickWrap) {
      const quickCandidates = FLAGSHIP_VISUAL_TASKS.filter((tid) => (taskMap.get(tid)?.gifCount || 0) > 0);
      const fallbackQuick = allTasks
        .filter((t) => t.gifCount > 0 && !quickCandidates.includes(t.taskId))
        .sort((a, b) => b.gifCount - a.gifCount || a.taskId.localeCompare(b.taskId))
        .map((t) => t.taskId);
      const pills = [...quickCandidates, ...fallbackQuick].slice(0, 10);
      if (pills.length > 0) {
        quickWrap.innerHTML =
          `<span class="gif-quick-label">Flagship visual tasks:</span>` +
          pills
            .map((tid) => {
              const cnt = taskMap.get(tid)?.gifCount || 0;
              const def = state.taskDefinitions.get(tid);
              const active = tid === state.gifTask ? "active" : "";
              return `<button type="button" class="gif-task-pill ${active}" data-quick-gif-task="${escapeHtml(tid)}" title="${escapeHtml(def?.title || tid)}"><code>${escapeHtml(tid)}</code> <span class="gif-pill-cnt">${cnt}</span></button>`;
            })
            .join("");
      } else {
        quickWrap.innerHTML = "";
      }
    }

    const top2Ids = new Set(ranked.slice(0, 2).map((f) => f.id));
    const top2Cov = new Map();
    for (const r of state.receipts) {
      if (r.gif_path && top2Ids.has(r.family_id)) {
        top2Cov.set(r.task_id, (top2Cov.get(r.task_id) || 0) + 1);
      }
    }

    let tasks = [...allTasks].sort((a, b) => {
      const aFlag = FLAGSHIP_VISUAL_TASKS.indexOf(a.taskId);
      const bFlag = FLAGSHIP_VISUAL_TASKS.indexOf(b.taskId);
      const aHas = a.gifCount > 0 ? 1 : 0;
      const bHas = b.gifCount > 0 ? 1 : 0;
      if (bHas !== aHas) return bHas - aHas;
      const aTop = top2Cov.get(a.taskId) || 0;
      const bTop = top2Cov.get(b.taskId) || 0;
      if (bTop !== aTop) return bTop - aTop;
      if (b.gifCount !== a.gifCount) return b.gifCount - a.gifCount;
      if ((aFlag !== -1) !== (bFlag !== -1)) return aFlag !== -1 ? -1 : 1;
      if (aFlag !== -1 && bFlag !== -1) return aFlag - bFlag;
      if (a.tier !== b.tier) return a.tier - b.tier;
      return a.taskId.localeCompare(b.taskId);
    });

    if (state.gifLawFilter) {
      tasks = tasks.filter((t) => taskMatchesLaw(state.taskDefinitions.get(t.taskId), state.gifLawFilter));
    }
    const q = state.gifSearch.trim().toLowerCase();
    if (q) {
      tasks = tasks.filter((t) => {
        const def = state.taskDefinitions.get(t.taskId);
        const hay = `${t.taskId} ${def?.title || ""} ${(def?.edge_cases || []).join(" ")}`.toLowerCase();
        return hay.includes(q);
      });
    }
    if (state.gifOnlyAvailable) {
      const withAny = tasks.filter((t) => t.gifCount > 0);
      if (withAny.length > 0 || q || state.gifLawFilter) {
        tasks = withAny;
      }
    }

    const taskSelect = $("gif-task");
    if (!taskSelect) return;
    if (tasks.length === 0) {
      taskSelect.innerHTML = `<option value="">No matching tasks</option>`;
      state.gifTask = "";
    } else {
      taskSelect.innerHTML = tasks
        .map((t) => {
          const gifTag = t.gifCount > 0 ? ` [${t.gifCount}/${state.families.length} GIFs]` : " [no GIF yet]";
          const tierTag = t.tier > 0 ? `T${t.tier} · ` : "";
          return `<option value="${escapeHtml(t.taskId)}">${escapeHtml(tierTag + t.taskId + gifTag)}</option>`;
        })
        .join("");
      if (!tasks.some((t) => t.taskId === state.gifTask)) {
        state.gifTask = pickDefaultGifTask(tasks, ranked, state.receipts);
        snapFamiliesToBest = true;
      }
      taskSelect.value = state.gifTask;
    }

    // Build family selector options sorted by leaderboard rank, with GIF & status indicators for current task
    const receiptByFam = new Map();
    for (const r of state.receipts) {
      if (r.task_id === state.gifTask) receiptByFam.set(r.family_id, r);
    }

    const famOptions = ranked
      .map((f, idx) => {
        const rec = receiptByFam.get(f.id);
        const st = rec ? rec.status : "not_run";
        const gifDot = rec && rec.gif_path ? "● " : "○ ";
        const score = `${f.full_pass}/${f.full_total}`;
        return `<option value="${escapeHtml(f.id)}">${escapeHtml(`${gifDot}#${idx + 1} ${f.label} (${score}) · ${st}`)}</option>`;
      })
      .join("");
    $("gif-family-left").innerHTML = famOptions;
    $("gif-family-right").innerHTML = famOptions;

    const bestPair = pickDefaultGifFamilies(state.gifTask, ranked, state.receipts, 2);
    if (snapFamiliesToBest || !ranked.some((f) => f.id === state.gifLeft)) {
      state.gifLeft = bestPair[0] || (ranked[0] && ranked[0].id) || "";
    }
    if (snapFamiliesToBest || !ranked.some((f) => f.id === state.gifRight) || state.gifRight === state.gifLeft) {
      state.gifRight =
        bestPair.find((id) => id !== state.gifLeft) ||
        (ranked.find((f) => f.id !== state.gifLeft) || ranked[0] || {}).id ||
        "";
    }
    $("gif-family-left").value = state.gifLeft;
    $("gif-family-right").value = state.gifRight;

    // Keep quick-task active pill synced
    if (quickWrap) {
      quickWrap.querySelectorAll("[data-quick-gif-task]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.quickGifTask === state.gifTask);
      });
    }
  }

  function renderGifCard(family, receipt, taskIdOverride = null, { rank = null, slotTag = null } = {}) {
    const taskId = taskIdOverride || state.gifTask;
    const statusHtml = receipt ? statusBadgeHtml(receipt.status) : `<span class="badge badge-not_run">not_run</span>`;
    const rawGifUrl = receipt && receipt.gif_path ? hfResolveUrl(receipt.gif_path) : null;
    const displayUrl = receipt && receipt.gif_path ? gifDisplayUrl(receipt.gif_path) : null;
    const famPage = githubBlobUrl(family.pagePath);
    const hashText = (receipt && receipt.source_hash) || family.sourceHash;
    const provText = formatGifProvenance(receipt);
    const rankHtml = rank != null ? `<span class="gif-rank-badge" title="Leaderboard full-suite rank">#${rank}</span>` : "";
    const slotHtml = slotTag ? `<span class="gif-slot-badge">${escapeHtml(slotTag)}</span>` : "";
    const scoreSub = `${family.full_pass}/${family.full_total} full-suite`;

    const mediaHtml = displayUrl
      ? `<div class="gif-img-wrap is-loading">
           <img src="${escapeHtml(displayUrl)}" alt="${escapeHtml(family.label)} training GIF for ${escapeHtml(taskId)}" loading="lazy" data-gif-img="1" data-gif-zoom-task="${escapeHtml(taskId)}" data-gif-zoom-fam="${escapeHtml(family.id)}">
           <button type="button" class="gif-zoom-btn" data-gif-zoom-task="${escapeHtml(taskId)}" data-gif-zoom-fam="${escapeHtml(family.id)}" title="Open fullscreen zoom inspector">⤢ Zoom</button>
         </div>`
      : `<div class="gif-placeholder">
           <strong>No training GIF published yet</strong>
           <span>Recorded status for <code>${escapeHtml(taskId)}</code>: ${escapeHtml(receipt ? receipt.status : "not_run")}.</span>
           ${receipt && receipt.reason ? `<div style="margin-top:6px">${escapeHtml(receipt.reason)}</div>` : ""}
         </div>`;

    return `
      <article class="gif-card ${displayUrl ? "has-gif" : "no-gif"}">
        <div class="gif-card-head">
          <div class="gif-card-head-left">
            <div class="gif-card-title-row">
              ${rankHtml}
              <span class="gif-card-title">${escapeHtml(family.label)}</span>
              ${slotHtml}
            </div>
            <div class="gif-card-sub"><code>${escapeHtml(family.candidateId || family.id)}</code> · ${escapeHtml(scoreSub)}</div>
          </div>
          ${statusHtml}
        </div>
        <div class="gif-media-box">${mediaHtml}</div>
        ${compactGifMetricsHtml(receipt)}
        <div class="gif-card-foot">
          <span class="gif-prov-text">${escapeHtml(provText || (hashText ? `src: ${shortHash(hashText, 10)}` : "Recorded cohort"))}</span>
          <span class="gif-card-links">
            ${rawGifUrl ? `<a href="${escapeHtml(rawGifUrl)}" target="_blank" rel="noopener noreferrer">Raw GIF ↗</a> · ` : ""}
            <a href="${escapeHtml(famPage)}" target="_blank" rel="noopener noreferrer">Report ↗</a>
          </span>
        </div>
      </article>
    `;
  }

  function attachGifHandlers(container) {
    if (!container) return;
    container.querySelectorAll("img[data-gif-img]").forEach((img) => {
      const wrap = img.closest(".gif-img-wrap");
      const markLoaded = () => {
        if (wrap) wrap.classList.remove("is-loading");
      };
      if (img.complete && img.naturalWidth > 0) {
        markLoaded();
      } else {
        img.addEventListener("load", markLoaded, { once: true });
      }
      img.addEventListener(
        "error",
        () => {
          const box = img.closest(".gif-media-box") || img.parentElement;
          if (!box) return;
          box.innerHTML = `
            <div class="gif-placeholder">
              <strong>GIF unavailable at this revision</strong>
              <span>Checked <code>${escapeHtml(img.getAttribute("src") || "")}</code></span>
            </div>`;
        },
        { once: true }
      );
    });
  }

  function openGifLightbox(taskId, familyId) {
    const { ranked } = buildFamilyRankMap(state.families);
    const byFam = new Map();
    for (const r of state.receipts) {
      if (r.task_id === taskId) byFam.set(r.family_id, r);
    }
    const availableIds = ranked.filter((f) => Boolean(byFam.get(f.id)?.gif_path)).map((f) => f.id);
    const familyIds = availableIds.includes(familyId) ? availableIds : [familyId, ...availableIds];
    const index = Math.max(0, familyIds.indexOf(familyId));
    state.gifLightbox = { taskId, familyIds, index };
    renderGifLightbox();
  }

  function closeGifLightbox() {
    state.gifLightbox = null;
    const lb = $("gif-lightbox");
    if (lb) lb.classList.add("hidden");
    if (typeof document !== "undefined" && !state.activeExperimentId) {
      document.body.classList.remove("modal-open");
    }
  }

  function renderGifLightbox() {
    const lb = $("gif-lightbox");
    if (!lb) return;
    if (!state.gifLightbox) {
      lb.classList.add("hidden");
      return;
    }
    const { taskId, familyIds, index } = state.gifLightbox;
    const { ranked, rankMap } = buildFamilyRankMap(state.families);
    const famId = familyIds[index];
    const fam = ranked.find((f) => f.id === famId);
    if (!fam) {
      closeGifLightbox();
      return;
    }
    const rec = state.receipts.find((r) => r.task_id === taskId && r.family_id === famId) || null;
    const def = state.taskDefinitions.get(taskId);
    const rawUrl = rec && rec.gif_path ? hfResolveUrl(rec.gif_path) : null;
    const dispUrl = rec && rec.gif_path ? gifDisplayUrl(rec.gif_path) : null;

    lb.classList.remove("hidden");
    document.body.classList.add("modal-open");

    $("gif-lb-title").textContent = `#${rankMap.get(famId) || "—"} ${fam.label} (${fam.candidateId || fam.id})`;
    $("gif-lb-task-id").textContent = `${taskId} · ${index + 1}/${familyIds.length}`;
    const prov = formatGifProvenance(rec);
    $("gif-lb-badges").innerHTML = [
      rec ? statusBadgeHtml(rec.status) : `<span class="badge badge-not_run">not_run</span>`,
      `<span class="badge badge-required">Full-Suite ${fam.full_pass}/${fam.full_total}</span>`,
      prov ? `<span class="badge badge-muted">${escapeHtml(prov)}</span>` : "",
      def && def.title ? `<span class="badge badge-evolving">${escapeHtml(def.title)}</span>` : "",
    ]
      .filter(Boolean)
      .join("");

    const rawLink = $("gif-lb-raw");
    if (rawLink) {
      rawLink.href = rawUrl || "#";
      rawLink.classList.toggle("hidden", !rawUrl);
    }
    if ($("gif-lb-prev")) $("gif-lb-prev").disabled = familyIds.length <= 1;
    if ($("gif-lb-next")) $("gif-lb-next").disabled = familyIds.length <= 1;

    $("gif-lb-media").innerHTML = dispUrl
      ? `<div class="gif-img-wrap is-loading"><img src="${escapeHtml(dispUrl)}" alt="${escapeHtml(fam.label)} training GIF for ${escapeHtml(taskId)}" data-gif-img="1"></div>`
      : `<div class="gif-placeholder"><strong>No GIF available</strong><span>Status: ${escapeHtml(rec ? rec.status : "not_run")}</span></div>`;
    attachGifHandlers($("gif-lb-media"));

    const metaParts = [];
    if (rec && rec.wall_seconds != null) metaParts.push(`Wall: ${rec.wall_seconds.toFixed(1)}s`);
    if (rec && rec.energy && rec.energy.joules_to_first_pass != null) {
      metaParts.push(`J-to-PASS: ${formatJoules(rec.energy.joules_to_first_pass)}`);
    }
    if (rec && rec.gif_meta && rec.gif_meta.attempt_id) {
      metaParts.push(`GIF attempt: ${shortHash(rec.gif_meta.attempt_id, 12)}`);
    }
    if (rec && rec.reason) metaParts.push(rec.reason);
    $("gif-lb-meta").innerHTML =
      compactGifMetricsHtml(rec) +
      `<div class="gif-lb-foot-note">${escapeHtml(metaParts.join(" · ") || "Use ← / → arrow keys to step across families, R to replay, Esc to close.")}</div>`;
  }

  function renderGifInspector() {
    const stage = $("gif-stage");
    if (!stage) return;
    const compareControls = state.gifLayout === "compare";
    const gridControls = state.gifLayout === "grid";
    $("gif-left-group").classList.toggle("hidden", !compareControls);
    $("gif-right-group").classList.toggle("hidden", !compareControls);
    $("gif-swap-group").classList.toggle("hidden", !compareControls);
    if ($("gif-hide-missing-group")) $("gif-hide-missing-group").classList.toggle("hidden", !gridControls);
    stage.className = compareControls
      ? "gif-stage gif-stage-compare"
      : state.gifLayout === "top4"
        ? "gif-stage gif-stage-top4"
        : "gif-stage gif-stage-grid";

    if (!state.gifTask) {
      $("gif-task-meta").textContent = "";
      if ($("gif-family-strip")) $("gif-family-strip").innerHTML = "";
      stage.innerHTML = `<div class="empty-cell">No task matches the current GIF filters.</div>`;
      return;
    }

    const { ranked, rankMap } = buildFamilyRankMap(state.families);
    const byFamily = new Map();
    for (const r of state.receipts) {
      if (r.task_id === state.gifTask) byFamily.set(r.family_id, r);
    }

    const def = state.taskDefinitions.get(state.gifTask);
    const gifAvail = ranked.filter((f) => Boolean(byFamily.get(f.id)?.gif_path)).length;
    const passCount = ranked.filter((f) => byFamily.get(f.id)?.status === "PASS").length;
    const failCount = ranked.filter((f) => byFamily.get(f.id)?.status === "FAIL").length;
    const probLink = `${githubBlobUrl("reports/forge/problems.md")}#${encodeURIComponent(state.gifTask)}`;
    $("gif-task-meta").innerHTML = `
      <button type="button" class="btn btn-ghost btn-sm" data-open-experiment="${escapeHtml(state.gifTask)}">Deep-Dive: <code>${escapeHtml(state.gifTask)}</code></button>
      <span><strong>${gifAvail}/${ranked.length}</strong> families with GIF · ${passCount} PASS · ${failCount} FAIL ·</span>
      <a href="${escapeHtml(probLink)}" target="_blank" rel="noopener noreferrer">Problem spec ↗</a>
      ${def && def.intuition ? `<div class="gif-task-intuition">${escapeHtml(def.intuition)}</div>` : ""}
    `;

    // Interactive Family Coverage Strip (leaderboard-ranked)
    const strip = $("gif-family-strip");
    if (strip) {
      strip.innerHTML = ranked
        .map((f, idx) => {
          const rec = byFamily.get(f.id);
          const hasGif = Boolean(rec && rec.gif_path);
          const status = rec ? rec.status : "not_run";
          const isLeft = compareControls && f.id === state.gifLeft;
          const isRight = compareControls && f.id === state.gifRight;
          const slotMark = isLeft ? `<span class="chip-slot">L</span>` : isRight ? `<span class="chip-slot">R</span>` : "";
          const statusCls = status === "PASS" ? "chip-pass" : status === "FAIL" ? "chip-fail" : "chip-muted";
          const activeCls = isLeft || isRight ? "is-selected" : "";
          return `<button type="button" class="gif-fam-chip ${statusCls} ${hasGif ? "has-gif" : "no-gif"} ${activeCls}" data-gif-chip-fam="${escapeHtml(f.id)}" title="#${idx + 1} ${escapeHtml(f.label)} (${f.full_pass}/${f.full_total}) · ${escapeHtml(status)} · ${hasGif ? "Click to inspect/compare GIF" : "No GIF published yet"}">
            <span class="chip-dot" aria-hidden="true">${hasGif ? "●" : "○"}</span>
            <span class="chip-rank">#${idx + 1}</span>
            <span>${escapeHtml(f.label)}</span>
            ${slotMark}
          </button>`;
        })
        .join("");
    }

    let selectedFamilies = [];
    if (compareControls) {
      const leftFam = ranked.find((f) => f.id === state.gifLeft);
      const rightFam = ranked.find((f) => f.id === state.gifRight);
      const cards = [];
      if (leftFam) cards.push(renderGifCard(leftFam, byFamily.get(leftFam.id), state.gifTask, { rank: rankMap.get(leftFam.id), slotTag: "Left" }));
      if (rightFam) cards.push(renderGifCard(rightFam, byFamily.get(rightFam.id), state.gifTask, { rank: rankMap.get(rightFam.id), slotTag: "Right" }));
      stage.innerHTML = cards.join("");
      attachGifHandlers(stage);
      return;
    }

    if (state.gifLayout === "top4") {
      const withGif = ranked.filter((f) => Boolean(byFamily.get(f.id)?.gif_path));
      const withoutGif = ranked.filter((f) => !byFamily.get(f.id)?.gif_path);
      selectedFamilies = [...withGif, ...withoutGif].slice(0, 4);
      stage.innerHTML = selectedFamilies
        .map((f) => renderGifCard(f, byFamily.get(f.id), state.gifTask, { rank: rankMap.get(f.id) }))
        .join("");
      attachGifHandlers(stage);
      return;
    }

    // All Families Grid: ordered by GIF availability first, then strictly leaderboard performance rank
    const withGif = ranked.filter((f) => Boolean(byFamily.get(f.id)?.gif_path));
    const withoutGif = ranked.filter((f) => !byFamily.get(f.id)?.gif_path);
    const rendered = withGif.map((f) => renderGifCard(f, byFamily.get(f.id), state.gifTask, { rank: rankMap.get(f.id) }));
    if (!state.gifHideMissingInGrid && withoutGif.length > 0) {
      for (const f of withoutGif) {
        rendered.push(renderGifCard(f, byFamily.get(f.id), state.gifTask, { rank: rankMap.get(f.id) }));
      }
    }
    stage.innerHTML = rendered.length > 0 ? rendered.join("") : `<div class="empty-cell">No families have a published GIF for this task yet.</div>`;
    attachGifHandlers(stage);
  }

  /* ---------- Dedicated Experiment Deep-Dive Modal (#experiment/<task_id>) ---------- */

  function parseExperimentHash(hashStr) {
    const m = /^#experiment\/([A-Za-z0-9_.\-]+)$/.exec(String(hashStr || "").trim());
    return m ? decodeURIComponent(m[1]) : null;
  }

  function openExperimentModal(taskId, { pushHash = true } = {}) {
    if (!taskId) return;
    state.activeExperimentId = taskId;
    const { ranked } = buildFamilyRankMap(state.families);
    state.expModalGifFamilies = pickDefaultGifFamilies(taskId, ranked, state.receipts, 2);

    if (pushHash && typeof window !== "undefined" && window.location) {
      const targetHash = `#experiment/${encodeURIComponent(taskId)}`;
      if (window.location.hash !== targetHash) {
        try {
          window.history.replaceState(null, "", targetHash);
        } catch {
          window.location.hash = targetHash;
        }
      }
    }
    renderExperimentModal();
  }

  function closeExperimentModal({ clearHash = true } = {}) {
    state.activeExperimentId = null;
    const modal = $("experiment-modal");
    if (modal) modal.classList.add("hidden");
    if (typeof document !== "undefined") document.body.classList.remove("modal-open");
    if (clearHash && typeof window !== "undefined" && window.location && parseExperimentHash(window.location.hash)) {
      try {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
      } catch {
        window.location.hash = "";
      }
    }
  }

  function renderExperimentModal() {
    const modal = $("experiment-modal");
    if (!modal) return;
    const taskId = state.activeExperimentId;
    if (!taskId) {
      modal.classList.add("hidden");
      document.body.classList.remove("modal-open");
      return;
    }

    const def =
      state.taskDefinitions.get(taskId) ||
      normalizeTaskDefinition({ task_id: taskId, tier: 0, role: "required" });
    const receiptsForTask = state.receipts.filter((r) => r.task_id === taskId);
    const receiptByFam = new Map(receiptsForTask.map((r) => [r.family_id, r]));

    modal.classList.remove("hidden");
    document.body.classList.add("modal-open");

    $("exp-modal-title").textContent = def.title || taskId;
    $("exp-modal-task-id").textContent = taskId;
    $("exp-modal-portal-link").href = def.pages_url || pagesUrl(`experiments/${taskId}/`);
    $("exp-modal-prob-link").href = `${githubBlobUrl("reports/forge/problems.md")}#${encodeURIComponent(taskId)}`;

    const jsonLink = $("exp-modal-json-link");
    if (def.task_config_path) {
      jsonLink.href = githubBlobUrl(def.task_config_path);
      jsonLink.classList.remove("hidden");
    } else {
      jsonLink.classList.add("hidden");
    }

    const tierBadge = def.tier > 0 ? `<span class="badge badge-required">Tier ${def.tier}</span>` : "";
    const lawBadges = (def.laws || [])
      .map(
        (lid) =>
          `<button type="button" class="badge badge-law" data-modal-filter-law="${escapeHtml(lid)}" title="${escapeHtml(lawFullTitle(lid))}">${escapeHtml(lawDisplayLabel(lid))}</button>`
      )
      .join("");
    const edgeBadges = (def.edge_cases || [])
      .map(
        (ec) =>
          `<button type="button" class="badge badge-edge" data-modal-filter-edge="${escapeHtml(ec)}">${escapeHtml(ec)}</button>`
      )
      .join("");
    $("exp-modal-badges").innerHTML = `${tierBadge}${roleBadgeHtml(def.role)}${lawBadges}${edgeBadges}`;

    // Target-Geometry SVG Schematic
    const safeSvg = sanitizeInlineSvg(def.svg_inline);
    const svgHtml = safeSvg
      ? safeSvg
      : `<div class="gif-placeholder">
           <strong>Target schematic pending SVG export</strong>
           <span>Expected schematic asset: <code>${escapeHtml(def.svg_relpath || `reports/forge/experiments/assets/${taskId}.svg`)}</code></span>
         </div>`;

    // Mermaid Protocol Diagram
    const mermaidSrc =
      def.mermaid ||
      [
        "flowchart LR",
        `  P["Prior: ${(def.prior_summary || "Task Prior").replace(/["`<>]/g, "")}"] --> G["Model: ${(def.model_summary || def.adapter || "Generator & Critic").replace(/["`<>]/g, "")}"]`,
        `  G --> E["Numerical Gate (${def.steps ? `${def.steps} steps` : "Task budget"})"]`,
      ].join("\n");

    // Thresholds table
    const threshEntries = Object.entries(def.thresholds || {});
    const gatesHtml =
      threshEntries.length > 0
        ? `<div class="table-wrap"><table class="data-table gates-table">
            <thead><tr><th>Gate Metric</th><th>Declared Threshold</th></tr></thead>
            <tbody>${threshEntries
              .map(
                ([k, v]) =>
                  `<tr><td><code>${escapeHtml(k)}</code></td><td><code>${escapeHtml(Array.isArray(v) ? JSON.stringify(v) : String(v))}</code></td></tr>`
              )
              .join("")}</tbody>
          </table></div>`
        : `<p class="muted" style="margin:0;font-size:0.82rem">No scalar thresholds embedded in this snapshot. Inspect the Task JSON for complete evaluation rules.</p>`;

    // Cross-Family Recorded Outcomes table (ordered by leaderboard performance rank)
    const { ranked: rankedFams, rankMap: modalRankMap } = buildFamilyRankMap(state.families);
    const outcomeRows = rankedFams
      .map((fam) => {
        const rec = receiptByFam.get(fam.id);
        const status = rec ? rec.status : "not_run";
        const wall = rec && rec.wall_seconds != null ? `${rec.wall_seconds.toFixed(1)} s` : "—";
        const j2p =
          rec && rec.energy && rec.energy.joules_to_first_pass != null
            ? formatJoules(rec.energy.joules_to_first_pass)
            : "—";
        const reason = rec && rec.reason ? rec.reason : "—";
        const inPreview = state.expModalGifFamilies.includes(fam.id);
        return `
          <tr class="clickable ${inPreview ? "selected-row" : ""}" data-modal-outcome-fam="${escapeHtml(fam.id)}">
            <td><span class="gif-rank-badge">#${modalRankMap.get(fam.id)}</span> <strong>${escapeHtml(fam.label)}</strong> ${fam.candidateId ? `<code>${escapeHtml(fam.candidateId)}</code>` : ""}</td>
            <td>${statusBadgeHtml(status)}</td>
            <td class="num">${escapeHtml(wall)}</td>
            <td class="num">${escapeHtml(j2p)}</td>
            <td class="reason-cell">${escapeHtml(reason)}</td>
            <td class="num">
              <button type="button" class="btn btn-sm ${rec && rec.gif_path ? "btn-primary" : "btn-ghost"}" data-modal-gif-jump="${escapeHtml(fam.id)}">${rec && rec.gif_path ? "Preview GIF" : "Compare"}</button>
            </td>
          </tr>`;
      })
      .join("");

    // Embedded side-by-side training GIF preview cards
    const previewFams = state.expModalGifFamilies
      .map((fid) => rankedFams.find((f) => f.id === fid))
      .filter(Boolean);
    const previewHtml =
      previewFams.length > 0
        ? previewFams.map((fam) => renderGifCard(fam, receiptByFam.get(fam.id), taskId, { rank: modalRankMap.get(fam.id) })).join("")
        : `<div class="empty-cell">Click any family row above to preview its training GIF here.</div>`;

    const body = $("exp-modal-body");
    body.innerHTML = `
      <div class="exp-context-grid">
        <div class="exp-context-card">
          <h3>What It Tests (Intuition)</h3>
          <p>${escapeHtml(def.intuition || "See problem specification and task contract for full details.")}</p>
        </div>
        <div class="exp-context-card">
          <h3>Common Failure Mode</h3>
          <p>${escapeHtml(def.failure_mode || "Naive or incomplete formulations lose modes, oscillate, or diverge under the declared task contract.")}</p>
        </div>
        <div class="exp-context-card">
          <h3>Protocol &amp; Serving Contract</h3>
          <p>${escapeHtml(def.protocol_notes || "Deterministic task-owned initializer, fixed budget, and live evaluation gate.")}</p>
        </div>
      </div>

      <div class="exp-visuals-grid">
        <div class="exp-visual-card">
          <div class="exp-visual-head">
            <h3>Target Geometry Schematic</h3>
            <span class="badge badge-evolving">Target Schematic — not training output</span>
          </div>
          <div class="svg-schematic-frame">${svgHtml}</div>
          <div class="schematic-caption">
            <span>Target Schematic — not training output</span>
            ${def.svg_relpath ? `<code>${escapeHtml(def.svg_relpath)}</code>` : ""}
          </div>
        </div>

        <div class="exp-visual-card">
          <div class="exp-visual-head">
            <h3>Protocol &amp; Pipeline Diagram</h3>
            <span class="badge badge-muted">Mermaid</span>
          </div>
          <div class="mermaid-host" data-mermaid-source="${escapeHtml(mermaidSrc)}">
            <pre class="mermaid-fallback">${escapeHtml(mermaidSrc)}</pre>
          </div>
        </div>
      </div>

      <div>
        <h3 class="exp-section-title">Architecture, Prior Contract &amp; Numerical Gates</h3>
        <div class="detail-meta-grid" style="margin-bottom:12px">
          <div class="meta-box">
            <div class="meta-box-label">Prior Contract</div>
            <div class="meta-box-value">${escapeHtml(def.prior_summary || "Task-owned prior")}</div>
          </div>
          <div class="meta-box">
            <div class="meta-box-label">Model / Host Summary</div>
            <div class="meta-box-value">${escapeHtml(def.model_summary || def.adapter || "Task-owned host")}</div>
          </div>
          <div class="meta-box">
            <div class="meta-box-label">Budget / Sampling</div>
            <div class="meta-box-value">${escapeHtml([def.steps != null ? `${def.steps} steps` : null, def.sampling_law || null, def.scoring_weights ? `weights: ${def.scoring_weights}` : null].filter(Boolean).join(" · ") || "Task-declared")}</div>
          </div>
          <div class="meta-box">
            <div class="meta-box-label">Views</div>
            <div class="meta-box-value">${escapeHtml(def.views.length > 0 ? def.views.join(", ") : "Full-suite / diagnostic")}</div>
          </div>
        </div>
        ${gatesHtml}
      </div>

      <div>
        <div class="subsection-head">
          <h3 class="exp-section-title" style="margin:0">Cross-Family Recorded Outcomes</h3>
          <button type="button" class="btn btn-ghost btn-sm" id="exp-modal-open-gif-inspector">Open in Full GIF Inspector →</button>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Family</th>
                <th>Status</th>
                <th class="num">Wall Time</th>
                <th class="num">J-to-PASS</th>
                <th>Reason / Note</th>
                <th class="num">GIF</th>
              </tr>
            </thead>
            <tbody>${outcomeRows}</tbody>
          </table>
        </div>
      </div>

      <div>
        <h3 class="exp-section-title">Side-by-Side Training GIF Preview</h3>
        <div class="gif-stage gif-stage-compare" id="exp-modal-gif-stage">${previewHtml}</div>
      </div>

      <div>
        <h3 class="exp-section-title">Reproduce This Experiment</h3>
        <div class="repro-bar">
          <code>${escapeHtml(def.reproduction_cmd)}</code>
          <button type="button" class="btn btn-primary btn-sm" id="exp-modal-copy-cmd">Copy CLI</button>
        </div>
      </div>
    `;

    attachGifHandlers($("exp-modal-gif-stage"));
    renderMermaidBlocks(body);

    const copyCmdBtn = $("exp-modal-copy-cmd");
    if (copyCmdBtn) {
      copyCmdBtn.addEventListener("click", () => copyTextWithFeedback(def.reproduction_cmd, copyCmdBtn));
    }
    const openGifBtn = $("exp-modal-open-gif-inspector");
    if (openGifBtn) {
      openGifBtn.addEventListener("click", () => {
        closeExperimentModal();
        state.gifTask = taskId;
        if (state.expModalGifFamilies[0]) state.gifLeft = state.expModalGifFamilies[0];
        if (state.expModalGifFamilies[1]) state.gifRight = state.expModalGifFamilies[1];
        state.gifLawFilter = "";
        if ($("gif-law-filter")) $("gif-law-filter").value = "";
        state.gifOnlyAvailable = false;
        if ($("gif-only-available")) $("gif-only-available").checked = false;
        populateGifSelectors();
        renderGifInspector();
        $("gif-section").scrollIntoView({ behavior: "smooth" });
      });
    }
  }

  /* ---------- Provenance Section ---------- */

  function renderProvenanceSection() {
    const m = state.meta;
    const dsViewer = m.datasetRepo === "local" ? resolveDataUrl() : `https://huggingface.co/datasets/${m.datasetRepo}`;
    const rawJson = isLocalMode() ? resolveDataUrl() : hfResolveUrl(CONFIG.dataPath, m.hfRevision);
    const declaredViewsStr = Object.entries(m.declaredViewRevisions || {})
      .map(([k, v]) => `${k} rev ${v}`)
      .join(", ");
    const cards = [
      ["Research Portal (GitHub Pages)", `<a href="${escapeHtml(pagesUrl(""))}" target="_blank" rel="noopener noreferrer">${escapeHtml(pagesUrl(""))} ↗</a>`],
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
    renderLawsSection();
    renderEnergyTable();
    renderEnergyDetail();
    populateFilterOptions();
    renderExplorer();
    renderReceiptDetail();
    populateGifSelectors();
    renderGifInspector();
    renderProvenanceSection();
    if (state.activeExperimentId) {
      renderExperimentModal();
    }
  }

  /* ---------- Data Loading ---------- */

  async function fetchHfDatasetSha() {
    if (isLocalMode()) return null;
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
    const dataUrl = resolveDataUrl();
    const label = isLocalMode() ? `local snapshot (${CONFIG.dataPath})` : `${CONFIG.datasetRepo} (${CONFIG.requestedRev})`;
    setStatus(`Loading leaderboard from ${label}…`, "loading");
    try {
      const [hfSha, dataResp] = await Promise.all([
        fetchHfDatasetSha(),
        fetch(dataUrl, { headers: { Accept: "application/json" }, cache: "no-cache" }),
      ]);

      if (!dataResp.ok) {
        throw new Error(`HTTP ${dataResp.status} fetching ${dataUrl}`);
      }

      const raw = await dataResp.json();
      normalizeDataset(raw, hfSha);
      setStatus(null);
      renderAll();

      // Open Experiment Deep-Dive if loaded with #experiment/<task_id>
      if (typeof window !== "undefined" && window.location) {
        const hashTask = parseExperimentHash(window.location.hash);
        if (hashTask) openExperimentModal(hashTask, { pushHash: false });
      }
    } catch (err) {
      setStatus(
        `Unable to load data from ${dataUrl}: ${err.message}. Check that data/leaderboard.json is available or retry.`,
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
    if ($("btn-theme")) $("btn-theme").textContent = THEME_LABELS[next];
    try {
      localStorage.setItem("particlegan_lb_theme", next);
    } catch {
      // ignore storage restrictions inside sandboxed iframes
    }
    // Re-render Mermaid diagrams cleanly in the new theme
    if (state.raw) {
      renderFamilyFormulationSection();
      if (state.activeExperimentId) renderExperimentModal();
    }
  }

  function jumpToExplorerWithFilters({ family = null, law = null, edgeCase = null } = {}) {
    if (family !== null) {
      state.expFilters.family = family;
      if ($("filter-family")) $("filter-family").value = family;
    }
    if (law !== null) {
      state.expFilters.law = law;
      if ($("filter-law")) $("filter-law").value = law;
    }
    if (edgeCase !== null) {
      state.expFilters.edgeCase = edgeCase;
      if ($("filter-edge-case")) $("filter-edge-case").value = edgeCase;
    }
    state.expLimit = 100;
    renderExplorer();
    if ($("explorer-section")) $("explorer-section").scrollIntoView({ behavior: "smooth" });
  }

  function wireEvents() {
    // Theme toggle
    let savedTheme = "auto";
    const themeParam = params.get("theme");
    if (themeParam && THEMES.includes(themeParam)) {
      savedTheme = themeParam;
    } else {
      try {
        savedTheme = localStorage.getItem("particlegan_lb_theme") || "auto";
      } catch {
        savedTheme = "auto";
      }
    }
    applyTheme(savedTheme);

    $("btn-theme").addEventListener("click", () => {
      const current = document.documentElement.dataset.theme || "auto";
      const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
      applyTheme(next);
    });

    $("btn-refresh").addEventListener("click", () => loadData());

    // Section nav active state
    document.querySelectorAll(".section-nav .nav-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        document.querySelectorAll(".section-nav .nav-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
      });
    });

    // Hash routing for #experiment/<task_id>
    window.addEventListener("hashchange", () => {
      const hashTask = parseExperimentHash(window.location.hash);
      if (hashTask) {
        openExperimentModal(hashTask, { pushHash: false });
      } else if (state.activeExperimentId) {
        closeExperimentModal({ clearHash: false });
      }
    });

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
      jumpToExplorerWithFilters({ family: state.selectedFamilyId });
    });

    if ($("fd-laws-btn")) {
      $("fd-laws-btn").addEventListener("click", () => {
        if (!state.selectedFamilyId) return;
        state.selectedFormulationFamilyId = state.selectedFamilyId;
        renderFamilyFormulationSection();
        $("laws-section").scrollIntoView({ behavior: "smooth" });
      });
    }

    // Laws & Families Matrix interactions
    $("laws-section").addEventListener("click", (e) => {
      const lawJump = e.target.closest("[data-filter-law-jump]");
      if (lawJump) {
        jumpToExplorerWithFilters({ law: lawJump.dataset.filterLawJump });
        return;
      }
      const heatBtn = e.target.closest("button[data-law-cell-family]");
      if (heatBtn) {
        const familyId = heatBtn.dataset.lawCellFamily;
        const lawId = heatBtn.dataset.lawCellLaw;
        const same =
          state.selectedLawCell &&
          state.selectedLawCell.familyId === familyId &&
          state.selectedLawCell.lawId === lawId;
        state.selectedLawCell = same ? null : { familyId, lawId };
        renderLawsSection();
        return;
      }
      const pill = e.target.closest("button[data-formulation-family]");
      if (pill) {
        state.selectedFormulationFamilyId = pill.dataset.formulationFamily;
        renderFamilyFormulationSection();
      }
    });

    if ($("lcd-close-btn")) {
      $("lcd-close-btn").addEventListener("click", () => {
        state.selectedLawCell = null;
        renderLawsSection();
      });
    }
    if ($("lcd-filter-tasks-btn")) {
      $("lcd-filter-tasks-btn").addEventListener("click", () => {
        if (!state.selectedLawCell) return;
        jumpToExplorerWithFilters({
          family: state.selectedLawCell.familyId,
          law: state.selectedLawCell.lawId,
        });
      });
    }
    if ($("lcd-open-family-btn")) {
      $("lcd-open-family-btn").addEventListener("click", () => {
        if (!state.selectedLawCell) return;
        state.selectedFormulationFamilyId = state.selectedLawCell.familyId;
        renderFamilyFormulationSection();
        $("family-formulation-block").scrollIntoView({ behavior: "smooth" });
      });
    }

    // Pareto chart controls
    document.querySelectorAll("[data-pareto-x]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.paretoX = btn.dataset.paretoX;
        renderParetoChart(getFilteredEnergyBenchmarks());
      });
    });
    document.querySelectorAll("[data-pareto-y]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.paretoY = btn.dataset.paretoY;
        renderParetoChart(getFilteredEnergyBenchmarks());
      });
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
    $("filter-law").addEventListener("change", (e) => {
      state.expFilters.law = e.target.value;
      state.expLimit = 100;
      renderExplorer();
    });
    $("filter-edge-case").addEventListener("change", (e) => {
      state.expFilters.edgeCase = e.target.value;
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
      state.expFilters = { family: "", law: "", edgeCase: "", tier: "", status: "", role: "", search: "", hasGif: false };
      $("filter-family").value = "";
      $("filter-law").value = "";
      $("filter-edge-case").value = "";
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

    // Explorer row click + Experiment Deep-Dive + Law inline badge + GIF jump button
    $("explorer-tbody").addEventListener("click", (e) => {
      const expBtn = e.target.closest("[data-open-experiment]");
      if (expBtn) {
        e.stopPropagation();
        openExperimentModal(expBtn.dataset.openExperiment);
        return;
      }
      const inlineLaw = e.target.closest("[data-filter-law-inline]");
      if (inlineLaw) {
        e.stopPropagation();
        jumpToExplorerWithFilters({ law: inlineLaw.dataset.filterLawInline });
        return;
      }
      const jumpBtn = e.target.closest("button[data-gif-jump]");
      if (jumpBtn) {
        e.stopPropagation();
        state.gifTask = jumpBtn.dataset.gifJump;
        state.gifLeft = jumpBtn.dataset.gifFam;
        state.gifLawFilter = "";
        if ($("gif-law-filter")) $("gif-law-filter").value = "";
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

    if ($("rd-open-exp-btn")) {
      $("rd-open-exp-btn").addEventListener("click", () => {
        const r = state.receipts.find((item) => item.key === state.selectedReceiptKey);
        if (r) openExperimentModal(r.task_id);
      });
    }

    $("rd-open-gif-btn").addEventListener("click", () => {
      const r = state.receipts.find((item) => item.key === state.selectedReceiptKey);
      if (!r) return;
      state.gifTask = r.task_id;
      state.gifLeft = r.family_id;
      state.gifLawFilter = "";
      if ($("gif-law-filter")) $("gif-law-filter").value = "";
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

    if ($("gif-law-filter")) {
      $("gif-law-filter").addEventListener("change", (e) => {
        state.gifLawFilter = e.target.value;
        populateGifSelectors();
        renderGifInspector();
      });
    }
    if ($("gif-search")) {
      $("gif-search").addEventListener("input", (e) => {
        state.gifSearch = e.target.value;
        populateGifSelectors();
        renderGifInspector();
      });
    }
    $("gif-task").addEventListener("change", (e) => {
      state.gifTask = e.target.value;
      const byFam = new Map(state.receipts.filter((r) => r.task_id === state.gifTask).map((r) => [r.family_id, r]));
      const leftMissing = !byFam.get(state.gifLeft)?.gif_path;
      const rightMissing = !byFam.get(state.gifRight)?.gif_path;
      populateGifSelectors({ snapFamiliesToBest: leftMissing && rightMissing });
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
    if ($("gif-hide-missing")) {
      $("gif-hide-missing").addEventListener("change", (e) => {
        state.gifHideMissingInGrid = e.target.checked;
        renderGifInspector();
      });
    }
    if ($("gif-quick-tasks")) {
      $("gif-quick-tasks").addEventListener("click", (e) => {
        const btn = e.target.closest("[data-quick-gif-task]");
        if (!btn) return;
        state.gifTask = btn.dataset.quickGifTask;
        state.gifLawFilter = "";
        state.gifSearch = "";
        if ($("gif-law-filter")) $("gif-law-filter").value = "";
        if ($("gif-search")) $("gif-search").value = "";
        populateGifSelectors({ snapFamiliesToBest: true });
        renderGifInspector();
      });
    }
    if ($("btn-gif-top2")) {
      $("btn-gif-top2").addEventListener("click", () => {
        const { ranked } = buildFamilyRankMap(state.families);
        const pair = pickDefaultGifFamilies(state.gifTask, ranked, state.receipts, 2);
        if (pair[0]) state.gifLeft = pair[0];
        if (pair[1]) state.gifRight = pair[1];
        state.gifLayout = "compare";
        document.querySelectorAll("[data-gif-layout]").forEach((b) => b.classList.toggle("active", b.dataset.gifLayout === "compare"));
        populateGifSelectors();
        renderGifInspector();
      });
    }
    if ($("btn-gif-contrast")) {
      $("btn-gif-contrast").addEventListener("click", () => {
        const { ranked } = buildFamilyRankMap(state.families);
        const pair = pickContrastGifPair(state.gifTask, ranked, state.receipts);
        if (pair[0]) state.gifLeft = pair[0];
        if (pair[1]) state.gifRight = pair[1];
        state.gifLayout = "compare";
        document.querySelectorAll("[data-gif-layout]").forEach((b) => b.classList.toggle("active", b.dataset.gifLayout === "compare"));
        populateGifSelectors();
        renderGifInspector();
      });
    }
    if ($("btn-gif-sync")) {
      $("btn-gif-sync").addEventListener("click", () => {
        state.gifSyncNonce += 1;
        renderGifInspector();
      });
    }
    if ($("gif-family-strip")) {
      $("gif-family-strip").addEventListener("click", (e) => {
        const chip = e.target.closest("[data-gif-chip-fam]");
        if (!chip) return;
        const fid = chip.dataset.gifChipFam;
        if (state.gifLayout === "compare") {
          if (fid === state.gifLeft || fid === state.gifRight) {
            openGifLightbox(state.gifTask, fid);
            return;
          }
          if (e.shiftKey || state.gifNextSlot === "right") {
            state.gifRight = fid;
            state.gifNextSlot = "left";
          } else {
            state.gifLeft = fid;
            state.gifNextSlot = "right";
          }
          populateGifSelectors();
          renderGifInspector();
        } else {
          openGifLightbox(state.gifTask, fid);
        }
      });
    }
    $("gif-stage").addEventListener("click", (e) => {
      const zoomEl = e.target.closest("[data-gif-zoom-task]");
      if (zoomEl) {
        openGifLightbox(zoomEl.dataset.gifZoomTask, zoomEl.dataset.gifZoomFam);
      }
    });
    $("gif-task-meta").addEventListener("click", (e) => {
      const expBtn = e.target.closest("[data-open-experiment]");
      if (expBtn) openExperimentModal(expBtn.dataset.openExperiment);
    });

    // Fullscreen GIF Lightbox events
    const lightbox = $("gif-lightbox");
    if (lightbox) {
      $("gif-lb-close").addEventListener("click", () => closeGifLightbox());
      lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox) closeGifLightbox();
      });
      $("gif-lb-prev").addEventListener("click", () => {
        if (!state.gifLightbox) return;
        const n = state.gifLightbox.familyIds.length;
        state.gifLightbox.index = (state.gifLightbox.index - 1 + n) % n;
        renderGifLightbox();
      });
      $("gif-lb-next").addEventListener("click", () => {
        if (!state.gifLightbox) return;
        const n = state.gifLightbox.familyIds.length;
        state.gifLightbox.index = (state.gifLightbox.index + 1) % n;
        renderGifLightbox();
      });
      $("gif-lb-replay").addEventListener("click", () => {
        state.gifSyncNonce += 1;
        renderGifLightbox();
      });
    }

    // Experiment Deep-Dive Modal interactions
    const modal = $("experiment-modal");
    if (modal) {
      $("exp-modal-close").addEventListener("click", () => closeExperimentModal());
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          closeExperimentModal();
          return;
        }
        const zoomEl = e.target.closest("[data-gif-zoom-task]");
        if (zoomEl) {
          openGifLightbox(zoomEl.dataset.gifZoomTask, zoomEl.dataset.gifZoomFam);
          return;
        }
        const lawBtn = e.target.closest("[data-modal-filter-law]");
        if (lawBtn) {
          closeExperimentModal();
          jumpToExplorerWithFilters({ law: lawBtn.dataset.modalFilterLaw });
          return;
        }
        const edgeBtn = e.target.closest("[data-modal-filter-edge]");
        if (edgeBtn) {
          closeExperimentModal();
          jumpToExplorerWithFilters({ edgeCase: edgeBtn.dataset.modalFilterEdge });
          return;
        }
        const famRow = e.target.closest("[data-modal-outcome-fam], [data-modal-gif-jump]");
        if (famRow) {
          const fid = famRow.dataset.modalGifJump || famRow.dataset.modalOutcomeFam;
          if (fid) {
            const next = [fid, ...state.expModalGifFamilies.filter((x) => x !== fid)].slice(0, 2);
            state.expModalGifFamilies = next;
            renderExperimentModal();
          }
        }
      });
      $("exp-modal-copy-link").addEventListener("click", (e) => {
        if (typeof window !== "undefined" && window.location) {
          copyTextWithFeedback(window.location.href, e.currentTarget);
        }
      });
      document.addEventListener("keydown", (e) => {
        if (state.gifLightbox) {
          if (e.key === "Escape") {
            closeGifLightbox();
          } else if (e.key === "ArrowLeft") {
            const n = state.gifLightbox.familyIds.length;
            state.gifLightbox.index = (state.gifLightbox.index - 1 + n) % n;
            renderGifLightbox();
          } else if (e.key === "ArrowRight") {
            const n = state.gifLightbox.familyIds.length;
            state.gifLightbox.index = (state.gifLightbox.index + 1) % n;
            renderGifLightbox();
          } else if (e.key === "r" || e.key === "R") {
            state.gifSyncNonce += 1;
            renderGifLightbox();
          }
          return;
        }
        if (e.key === "Escape" && state.activeExperimentId) {
          closeExperimentModal();
        }
      });
    }
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
      CONFIG,
      normalizeDataset,
      normalizeLaw,
      normalizeFamily,
      normalizeTaskDefinition,
      normalizeReceipt,
      normalizeEnergyBenchmark,
      normalizeEnergyObj,
      normalizeStatus,
      normalizeRole,
      normalizeGifPath,
      isLocalMode,
      resolveDataUrl,
      hfResolveUrl,
      formatJoules,
      taskMatchesLaw,
      buildParetoPoints,
      parseExperimentHash,
      getFilteredReceipts,
      sanitizeInlineSvg,
      rankFamiliesByPerformance,
      pickDefaultGifTask,
      pickDefaultGifFamilies,
      pickContrastGifPair,
    };
  }
})();
