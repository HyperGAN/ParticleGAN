"""Build the compact review catalog from pinned PRs and completed captures."""
import argparse
from collections import Counter
from pathlib import Path

from .capture import write
from .render import read

IMAGE_QUESTIONS = {
 170:"Left-heavy versus right-heavy raised-dot templates; tactile glyph asymmetry, not Braille decoding.",
 166:"Two fixed barcode-like templates with opposite quiet-zone placement; not barcode validity or decoding.",
 159:"Two fixed near/far echo-location templates; not acoustic propagation or range inference.",
 154:"Two fixed rising/falling spectrogram-like traces; not audio synthesis or frequency generalization.",
 151:"Two fixed menu-icon layouts; not UI interaction or semantic object recognition.",
 150:"Two fixed moiré/beat intensity patterns; not recovery of unseen frequencies or phase.",
 131:"Fixed open C versus closed O pixel templates; RMSE is not an independent topology oracle.",
 106:"Fixed play/pause icon templates; not video dynamics or button behavior.",
 80:"Two fixed grayscale traffic-stack patterns; not red/green color semantics or traffic rules.",
 79:"Two fixed diagonal finder layouts; not QR recognition or error correction.",
 78:"Fixed letterbox versus pillarbox border placement; not aspect-ratio inference from arbitrary images.",
 77:"Fixed bright-center versus dark-center radial intensity patterns; not shape-from-shading.",
 76:"Fixed mirrored b/d glyph templates; not general OCR or a dedicated chirality score.",
 75:"Fixed ascending/descending stair templates; not sequence reasoning.",
 74:"Fixed two-dot/three-dot templates; not counting arbitrary objects, positions or cardinalities.",
 73:"Fixed mirrored L templates; not chirality generalization.",
 72:"Fixed opposite-handed swirl patterns; not rotation dynamics or optical flow.",
 71:"Fixed center/edge focus profiles; not depth estimation or optics reconstruction.",
 70:"Fixed corner-ramp intensity templates; not a learned coordinate system.",
 69:"Fixed smile/frown arcs; not emotion classification or facial-image fidelity.",
 68:"Fixed T-junction patterns; not occlusion reasoning.",
 67:"Fixed radial/wedge patterns; not a segmentation or reconstruction task.",
 66:"Fixed foreground/background intensity inversions; not conditional image inversion.",
 65:"Fixed grayscale left/right intensity patterns; no color channels or conditional grayscale-to-color query.",
 64:"Fixed vertical/horizontal bars; same orientation-coverage question as the shipped stripes family.",
 63:"Fixed templates named mask-inpaint; no observed image or mask enters G, so no conditional inpainting is tested.",
 62:"Fixed diagonal intensity ramps; not arbitrary image-algebra operations.",
 61:"Fixed sparse-observation-like templates; no source-domain input or paired correspondence tests translation.",
 59:"Fixed soft radial/ring templates; not a continuous stochastic shape family.",
 58:"Shipped intensity2 data reused as an architecture counterexample; not an independent new problem.",
}

BEHAVIOR_QUESTIONS={
 "two_pole":"Checks nonzero travel and bounded median critic gradient; its gate does not require both poles or a correct distribution.",
 "trajectory":"Checks the extracted trajectory edit while preserving identity in finite paired rows.",
 "residual_student":"Checks whether the intended residual moves toward the correct paired target.",
 "unipolar":"Checks an intended edit with preservation of unrelated content.",
 "ae_gan_hold":"Checks reconstruction/identity and an acquired adversarial edit during the declared hold.",
 "cover_leftover":"Checks target coverage plus the separate unwanted-remainder/content constraints.",
 "unused_token_hold":"Checks that active controls move and unused controls remain unchanged.",
 "mid_scale_identity":"Checks identity preservation and target edit magnitude at intermediate control strength.",
 "mode_hold":"Checks all eight ring modes and HQ through the sampled terminal hold; not within-mode density fidelity.",
 "reserved_annulus":"Check rotationally symmetric annular support with uniform-in-area radial mass; formerly reserved, now seen audit data.",
 "img_residual_bars4":"Check four-position template coverage under residual upsampling; formerly reserved architecture, now seen audit data.",
 "reserved_alternating_critic_updates":"Check ring quality with D updating every other outer step; formerly reserved cadence, now seen audit data.",
}

SOURCE_QUESTIONS=[
 (4,"Sparse mixed identity symbols","Match conditional real modes, active noise, exact inactive zeros and deterministic class symbols; W1 alone misses zero leakage."),
 (4,"Sparse mixed split symbols","Match each class's 50/50 two-symbol law and continuous/symbol consistency, rather than one valid symbol."),
 (4,"Analytic denoising grid, one class","Match the exact multimodal q(x0|xt) posterior, rather than a posterior mean."),
 (4,"Analytic denoising grid, four classes","Match q(x0|xt,c) and checkerboard class consistency against the analytic posterior."),
 (4,"Two-route trajectories, discrete geometry","Match 0.8/0.3 class route probabilities, continuous variation, endpoints and segment-level obstacle clearance on held-out geometry."),
 (4,"Two-route trajectories, continuous geometry","Same conditional route/support question with continuously sampled training geometry and fixed held-out contexts."),
 (4,"Route transitions, discrete geometry","Match a joint state/action/next-state law and next=state+action, not only the three marginals."),
 (4,"Route transitions, continuous geometry","Match joint transition relationships under continuous geometry and held-out contexts; class/route marginals alone are insufficient."),
 (4,"Paired affine2 transport","Learn the analytic paired input-output affine mapping on held-out rows; distinguish correspondence from output-marginal matching."),
 (4,"Paired swirl2 transport","Learn the radius-dependent paired rotation on held-out rows; fixed/movable clouds are model controls, not new data laws."),
 (3,"Ring acquisition/hold/warm/cold/shift protocols","Retain all eight modes in an uninterrupted own-state continuation and reacquire after target shift; many solver PRs reuse this law."),
 (3,"Paired sign kinematic lander","Expose joint-law sign symmetry versus correct paired control; one scalar sign around a supplied expert, not YuE2 itself."),
 (3,"Safe-fast kinematic lander","Check safety/speed objective composition for a one-parameter sink controller, not full physical policy learning."),
 (3,"Native paired-action particle fixture","Check paired-reference and particle/controller plumbing on a small constructed action host."),
 (3,"E22 routed support/pair/moving/replay fixtures","Check each declared dense-bank routing, support/width, paired residual and own-state replay contract; distinct from independent-row Atlas."),
 (2,"Five-word latent autoencoder demo","Recover the five fixed words and reconstruction, not text-generation or unseen-word generalization."),
 (2,"Single Gaussian quickstart","Check training/sample plumbing for a unimodal law; weak as a broad multimodal quality benchmark."),
]


def base_rating(name):
    if name=="two_pole":return 2,"Well defined but weak as a density test"
    if name.startswith("stress_") or name.startswith("reserved_alternating"):
        return 3,"Well defined; existing ring stress, not a new data family"
    if name in ["img_mean_discriminator","img_uniform_generator"]:
        return 2,"Well defined impossible/nonidentifiable diagnostic; bad qualification gate"
    if name in ["img_bars8","img_tiny_generator","vector_narrow"]:
        return 3,"Well defined diagnostic; solvability/budget remains separate"
    if name in ["vector_unequal_mass","vector_unequal_width","vector_anisotropic"]:
        return 5,"Good independent density question"
    return 4,"Good bounded regression question"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prs",type=Path,required=True);ap.add_argument("--sources",type=Path,required=True)
    ap.add_argument("--artifacts",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--addenda",type=Path,help="Pinned supplemental reviews; preserve the original snapshot and cohort")
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    supplement=read(args.addenda) if args.addenda else {}
    prs=read(args.prs);by_pr={p["number"]:p for p in prs};cases=[];broken=[]
    for record in read(args.artifacts/"develop-frozen-reference/index.json"):
        score,quality=base_rating(record["name"])
        cases.append(dict(id="develop-"+record["name"],name=record["name"],origin="develop",rating=score,quality=quality,
                          verifies=BEHAVIOR_QUESTIONS.get(record["name"],record["spec"]["importance_reason"]),
                          sampling=record["sampling"],reference="frozen host recipe + published architecture",spec=record["spec"],
                          final_live=record["live"],status=record["verdict"]["status"],convergence=record["verdict"].get("convergence"),
                          media="media/develop-"+record["name"]+".gif",source_result_sha256=record["result_sha256"]))
    for path in sorted(args.artifacts.glob("pr*/capture-index.json")):
        records=read(path)
        if not records:continue
        n=int(path.parent.name.removeprefix("pr").removesuffix("-adapted"));spec=records[0]["spec"]
        kind=records[0]["kind"]
        score=4 if n==45 else 3
        quality="Well defined template regression" if kind=="image" else "Well defined but redundant/confounded scale counterexample"
        if n in [61,63,65]:score=2;quality="Bad as the implied conditional application; valid unconditional template sampler"
        if n==58:quality="Duplicate shipped data; useful architecture counterexample"
        if n==45:quality="Good bounded unit-change counterexample"
        question=IMAGE_QUESTIONS[n] if kind=="image" else spec["importance_reason"]
        replay=read(path.parent/"replay.json") if (path.parent/"replay.json").exists() else {}
        adapters=replay.get("adapter")
        arms=[dict(status=r["verdict"]["status"],spec=r["spec"],live=r["live"],ema=r["ema"],
                   convergence=r["verdict"].get("convergence"),capture_sha256=r["capture_sha256"],result_sha256=r["result_sha256"])
              for r in records]
        cases.append(dict(id=path.parent.name,name=records[0]["name"],origin=f"PR{n}",pr=n,head_sha=by_pr[n]["headRefOid"],
                          rating=score,quality=quality,verifies=question,reference="proposal's original arms; current develop host"+("; archived GAN-v3 factory" if adapters else ""),
                          sampling=records[0]["sampling"],spec=spec,arms=arms,status=" / ".join(r["status"] for r in arms),
                          adapter=adapters,replay={k:v for k,v in replay.items() if k in
                            ["status","proposal_exit_code","receipt_reconstructed","reconstruction_basis","original_source_sha256"]},
                          media="media/"+path.parent.name+".gif"))
        if adapters:
            raw=read(args.artifacts/f"pr{n}/replay.json")
            broken.append(dict(pr=n,type="historical public-API import",error=raw["error"].strip().splitlines()[-1],
                               current_replay="Explicit archived factory only; not current KA2/Atlas qualification"))
        if n in [70,76,154,63]:
            broken.append(dict(pr=n,type="historical FAIL/PASS contrast changed",current_status=cases[-1]["status"],
                               next_action="Keep exact results; refresh counterexample status before treating it as a gate. Config repairs belong to the separate workstream."))
    for summary in sorted(args.artifacts.glob("native-*/summary.json")):
        r=read(summary);moving="moving_original_status" in r
        cases.append(dict(id="atlas-"+r["name"],name=r["name"],origin="develop / PR223 reference",rating=4 if moving else 5,
                          quality="Good target-reacquisition diagnostic" if moving else "Good density fidelity benchmark",
                          verifies=("Reacquire a rotated 100-mode target after two 30-degree shifts; the original moving gate is weaker than full density fidelity."
                                    if moving else "Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws."),
                          reference="Atlas affine native fixture, one fixed seed, external full budget",sampling=r["sampling"],
                          spec=dict(recipe=r["recipe"],steps=r["steps"],seed=r["seed"],device=r["device"]),
                          status="moving "+r["moving_original_status"]+" / strict native "+r["native_noisy_status"]+" / accuracy "+r["accuracy_status"] if moving else "noisy "+r["native_noisy_status"]+" / clean "+r["native_clean_status"]+" / accuracy "+r["accuracy_status"],
                          final=r["final"],noisy_passing_suffix=r["noisy_passing_suffix"],capture_sha256=r["capture_sha256"],
                          media="media/atlas-"+r["name"]+".gif"))
    oracles=read(args.output/"misgan-oracles.json")["controls"]
    for path in sorted(args.artifacts.glob("pr196-*/run/summary.json")):
        r=read(path);name=path.parents[1].name;mech=r["config"]["mechanism"]
        cases.append(dict(id=name,name="MisGAN "+mech,origin="PR196",pr=196,head_sha=by_pr[196]["headRefOid"],rating=4,
                          quality="Good analytic conditional test; missing predeclared scalar pass gate",
                          verifies="Learn complete 8D data from independent missingness AND sample the conditional missing-coordinate posterior; compare ambiguous rows to the exact Bayes oracle.",
                          reference="Unchanged MisGAN proposal loop using current public recipe",sampling="EMA, clean data-generator draw; 16 stochastic imputations per fixed test row",
                          spec=r["config"],final=r["final"],oracle=oracles[mech],status="Measured; no declared binary gate",media="media/"+name+".gif"))
    stiff=read(args.artifacts/"pr224-v2/replay.json")
    cases.append(dict(id="pr224",name="Constructed stiff-game LR release",origin="PR224",pr=224,head_sha=by_pr[224]["headRefOid"],rating=5,
                      quality="Good causal controller unit fixture",verifies="One native SettleTest LR release can destabilize a settled stiff coordinate; cancel only that release and allow safe-geometry releases.",
                      reference="Native public E22 G optimizer / RpGAN / SettleTest; fixed constructed critic and specified reachable Adam snapshot",
                      sampling="Deterministic two-coordinate unit game; no dataset/prior",status="Counterexample reproduced; both controls stable",final=stiff["final"],media="media/pr224.gif"))
    for n,title,question in [(22,"Circle closed-loop controller","Learn context-dependent signed tangential motion and radial recovery from independent local rows; keep true-environment radius, direction and speed over 1024 closed-loop steps."),
                              (153,"Sprite world model","Predict one-step state+render, then free-run 5/20/50-step dreams against an exact simulator on held-out ID and ceiling-bounce OOD episodes.")]:
        raw=read(args.artifacts/f"pr{n}-current/replay.json");error=raw["error"].strip().splitlines()[-1]
        cases.append(dict(id=f"pr{n}",name=title,origin=f"PR{n}",pr=n,head_sha=by_pr[n]["headRefOid"],rating=4,quality="Well defined scientific question; current develop execution blocked",
                          verifies=question,status="BLOCKED before training",error=error,media=None,
                          historical_media="media/pr153-historical-rollout.gif" if n==153 else "media/pr22-historical-rollout.png"))
        broken.append(dict(pr=n,type="current-develop execution blocked",error=error,next_action="Provide a current API-compatible host and genuine checkpoint-over-training GIF; preserve the historical endpoint evidence separately."))
    for i,(rating,name,question) in enumerate(SOURCE_QUESTIONS):
        cases.append(dict(id=f"source-family-{i:02d}",name=name,origin="develop source review",rating=rating,
                          quality="Source-reviewed definition; no fresh current training evidence",verifies=question,
                          status="SOURCE REVIEW ONLY",media=None,explanation="EXISTING_FAMILIES.md"))
    cases.extend(supplement.get("cases",[]))
    cases.sort(key=lambda r:(-r["rating"],r["name"]))
    public_hashes=read(args.artifacts/"native-grid100-v2/summary.json")["source_sha256"]
    write(args.output/"catalog.json",dict(base_sha="4b16312e56328a679b92da69a287c0c9490259d9",date="2026-10-01",public_package_sha256=public_hashes,cases=cases,
          scope="Test definitions and explanations; no production config or trainer repairs; scientific ratings are not model pass rates",broken=broken))
    problems=["# Ranked toy problems","", "Scores rate scientific usefulness: **5** = precise discriminating question with strong oracle/controls; **4** = useful bounded question; **3** = well-defined but narrow/redundant; **2** = weak gate or unsupported application interpretation. A model FAIL does not lower a problem's scientific rating.","",
              "The table includes protocol/architecture variants for traceability. Similar templates, ring stresses and wide-gap polygons are not independent benchmark families. `PASS / FAIL` is the original counterexample arm followed by its proposed positive control. Read [HIGH_RATED.md](HIGH_RATED.md) for the strongest tests' exact claims.","",
              "| Score | Problem / source | What it actually verifies | Current evidence | Training visualization |","|---|---|---|---|---|"]
    if supplement:
        problems[6:6]=["[PR226/227 addendum](PR226_PR227.md) adds two retained-evidence reviews on pinned develop `6ec7e578`; their cohort remains separate from the original `4b16312e` runs.",""]
    for c in cases:
        source=f"[{c['origin']}](https://github.com/HyperGAN/ParticleGAN/pull/{c['pr']})" if c.get("pr") else c["origin"]
        media=f"[GIF]({c['media']})" if c.get("media") else "[Source review; no fresh GIF](EXISTING_FAMILIES.md)" if c["status"]=="SOURCE REVIEW ONLY" else "**Missing: execution blocked**"
        question=c["verifies"].replace("|",r"\|")
        problems.append(f"| {c['rating']} | {c['name']} · {source} | {question} | {c['status']} | {media} |")
    (args.output/"PROBLEMS.md").write_text("\n".join(problems)+"\n")
    new={c["pr"] for c in cases if c.get("pr")}
    inventory=[];lines=["# Open pull-request scope review","",f"Snapshot: **{len(prs)} open PRs**; every exact head/base pair has a complete local Python-file diff. Deep problem/evaluator/arm review and fresh replay cover the 47 problem/counterexample PRs; other rows are a title/body/file-scope review, not validation of those algorithms. PR223 is merged and serves as the presentation/reference example.","",
                         "| PR | Scope | Problem evidence / action | Pinned head |","|---|---|---|---|"]
    for r in prs:
        n=r["number"]
        if n in new:
            scope="Existing-data counterexample" if n==58 else "New test/counterexample proposal"
            matched=[c for c in cases if c.get("pr")==n]
            action="; ".join(c["status"] for c in matched)
        elif n==216:scope="28-unit runner migration";action="Existing definitions; does not establish new task solvability. Keep historical/current API and sampling cohorts separate."
        elif n==222:scope="PacGAN-8 method on existing grid100";action="Only a 4-update smoke is reported; no full 7k convergence evidence for this arm. Atlas's grid GIF does not certify PacGAN."
        elif n in [214,215,217]:scope="Candidate public recipe";action="Existing toy families; no fresh qualification or promotion claim in this review."
        elif n in [98,99]:scope="Existing native/ring host; changed objective/anchor";action="GH9 likelihood or balanced anchors change the solver/objective, not the target distribution. Existing-host GIFs do not certify these methods."
        else:scope="Method, initialization, controller or diagnostic on existing toys";action="No additional independent toy distribution found in added/modified Python paths and PR scope. See the existing problem-family explanations."
        inventory.append(dict(number=n,title=r["title"],url=r["url"],head_sha=r["headRefOid"],base_sha=r["baseRefOid"],base=r["baseRefName"],scope=scope,
                              action=action,python_files=r["code_files"],complete_local_diff=r["complete_local_diff"]))
        lines.append(f"| [{n}](https://github.com/HyperGAN/ParticleGAN/pull/{n}) {r['title'].replace('|','/')} | {scope} | {action} | `{r['headRefOid'][:12]}` |")
    if supplement:
        lines.extend(["", "## PR226/227 supplemental review", "",
                      "Added after the original 134-PR snapshot. Exact proposal heads and develop base `6ec7e578` are preserved separately; training artifacts were reused without retraining. See [the review](PR226_PR227.md).", "",
                      "| PR | Scope | Problem evidence / action | Pinned head |", "|---|---|---|---|"])
        for r in supplement.get("pull_requests",[]):
            inventory.append(r)
            lines.append(f"| [{r['number']}]({r['url']}) {r['title']} | {r['scope']} | {r['action']} | `{r['head_sha'][:12]}` |")
    write(args.output/"pull_requests.json",inventory)
    (args.output/"OPEN_PRS.md").write_text("\n".join(lines)+"\n")
    sources=read(args.sources/"receipts.json")
    for n,path in [(153,"reports/animation/leaderboard/base_dream.gif"),(22,"reports/circle-transition/radial_hold_val.png")]:
        import hashlib
        local=args.sources/str(n)/path
        if local.exists() and not any(s["pr"]==n and s["path"]==path for s in sources):
            sources.append(dict(pr=n,head_sha=by_pr[n]["headRefOid"],path=path,
                                sha256=hashlib.sha256(local.read_bytes()).hexdigest(),role="historical endpoint only"))
    sources.extend(supplement.get("source_receipts",[]))
    write(args.output/"source-receipts.json",[{k:v for k,v in s.items() if k!="local"} for s in sources])
    print(dict(cases=len(cases),new_problem_prs=len(new),open_prs=len(prs),rating_counts=dict(Counter(c["rating"] for c in cases))),flush=True)


if __name__=="__main__":main()
