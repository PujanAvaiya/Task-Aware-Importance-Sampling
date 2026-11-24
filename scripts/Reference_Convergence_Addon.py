# REFERENCE-CONVERGENCE ADD-ON FOR Task_Aware_VAE_Followup_Colab.ipynb
# Paste this complete cell AFTER Section 15 (seed-level analysis).
# Does not retrain VAEs. Uses the saved checkpoints and the frozen, already-selected proposals.
# Important: run the notebook's SETUP/FUNCTION cells first so that FC, cfg, FUOUT, etc. exist.

import itertools
from dataclasses import asdict

# Choose "pilot" first. It processes 5 seeds / 16 held-out images per dimension.
# Once checked, change to "full" to run all 25 seeds / 104 held-out images per dimension.
REF_AUDIT_MODE = globals().get("REF_AUDIT_MODE", "pilot")   # "pilot" or "full"
REFERENCE_SIZES = (32_768, 65_536, 131_072)  # draws PER independent reference (A and B)
REFERENCE_BOOTSTRAPS = 5000

if REF_AUDIT_MODE not in ("pilot", "full"):
    raise ValueError("REF_AUDIT_MODE must be 'pilot' or 'full'")
if FOLLOWUP_MODE != "full":
    raise RuntimeError("This add-on is for the full VAE follow-up, not the smoke test")
if any(n <= 0 or n % FC.reference_chunk for n in REFERENCE_SIZES):
    raise ValueError("Reference sizes must be positive multiples of FC.reference_chunk")
if len(set(REFERENCE_SIZES)) != len(REFERENCE_SIZES):
    raise ValueError("Reference sizes must be unique")
N_LIST = tuple(sorted(REFERENCE_SIZES))
SEED_COUNT = 5 if REF_AUDIT_MODE == "pilot" else FC.n_seeds
IMAGE_COUNT = 16 if REF_AUDIT_MODE == "pilot" else len(x_follow)
# Deterministically chosen ONCE, independent of the experimental outcomes.
image_rng = np.random.default_rng(20261123)
IMAGE_IDS = np.sort(image_rng.choice(len(x_follow), IMAGE_COUNT, replace=False))
AUDIT_SPEC = {
    "audit_mode": REF_AUDIT_MODE, "N_each": list(N_LIST),
    "seeds": SEED_COUNT, "image_ids_within_x_follow": IMAGE_IDS.tolist(),
    "online_repeats": FC.online_repeats, "online_average": FC.online_average,
    "reference_chunk": FC.reference_chunk,
    "orig_fuout": str(FUOUT), "followup_config": asdict(FC)
}
AUDIT_HASH = hashlib.sha256(json.dumps(AUDIT_SPEC, sort_keys=True).encode()).hexdigest()[:12]
REF_AUDIT_OUT = FUOUT / f'reference_convergence_{REF_AUDIT_MODE}_{AUDIT_HASH}'
REF_AUDIT_OUT.mkdir(parents=True, exist_ok=True)
(REF_AUDIT_OUT / 'audit_config.json').write_text(json.dumps(AUDIT_SPEC, indent=2))
print('Mode:', REF_AUDIT_MODE, '| output:', REF_AUDIT_OUT)
print('Reference draws per independent pass:', N_LIST)
print('Training seeds per dimension:', SEED_COUNT, '| images per seed:', IMAGE_COUNT)

@torch.inference_mode()
def reference_checkpoints(model, images, rng_seed, proposal, checkpoints):
    """One pass to max N; return SNIS mu/logZ/ESS snapshots at each N.

    Image batches and log-sum-exp stabilization keep GPU memory bounded.
    Each call has independent random draws when its rng_seed differs.
    """
    nmax = max(checkpoints)
    snapshots = {n: {'mu': [], 'logz': [], 'ess': []} for n in checkpoints}
    g = torch_rng(rng_seed)
    for begin in range(0, len(images), FC.ref_image_batch):
        xb = images[begin:begin + FC.ref_image_batch]
        batch_n, d = len(xb), model.latent_dim
        maximum = torch.full((batch_n,), -float('inf'), dtype=torch.float64, device=DEVICE)
        sw = torch.zeros(batch_n, dtype=torch.float64, device=DEVICE)
        sw2 = torch.zeros(batch_n, dtype=torch.float64, device=DEVICE)
        swz = torch.zeros((batch_n, d), dtype=torch.float64, device=DEVICE)
        for first in range(0, nmax, FC.reference_chunk):
            draw_n = min(FC.reference_chunk, nmax - first)
            z, lw = draw_proposal(model, xb, draw_n, g, proposal)
            z, lw = z.double(), lw.double()
            nxt = torch.maximum(maximum, lw.max(dim=1).values)
            scale = torch.exp(maximum - nxt)
            w = torch.exp(lw - nxt[:, None])
            sw = sw * scale + w.sum(1)
            swz = swz * scale[:, None] + (w[:, :, None] * z).sum(1)
            sw2 = sw2 * scale.square() + w.square().sum(1)
            maximum = nxt
            processed = first + draw_n
            if processed in snapshots:
                snapshots[processed]['mu'].append((swz / sw[:, None]).cpu().numpy().copy())
                snapshots[processed]['logz'].append((maximum + sw.log() - math.log(processed)).cpu().numpy().copy())
                snapshots[processed]['ess'].append((sw.square() / sw2).cpu().numpy().copy())
    return {n: {key: np.concatenate(parts) for key, parts in record.items()}
            for n, record in snapshots.items()}

@torch.inference_mode()
def online_estimates_from_frozen_allocation(model, selected, regressors, seed, image_ids):
    """Replays the two selected-proposal methods with the ORIGINAL full-test group budgets.

    We only *score* image_ids; allocation and RNG generation still cover all 104 images.
    This avoids changing the original test protocol by using a smaller allocation group.
    """
    feats = get_features(model, x_follow)
    score = np.exp(regressors['Task512'].predict(feats)).clip(1e-8, 1e8)
    n = len(x_follow)
    k_uniform = np.empty(n, dtype=np.int32)
    k_task = np.empty(n, dtype=np.int32)
    for start in range(0, n, FC.group_size):
        end = start + FC.group_size
        total = FC.group_size * FC.online_average
        k_uniform[start:end] = bounded_allocate(
            np.ones(FC.group_size), total, cfg.k_min, cfg.k_max)
        k_task[start:end] = bounded_allocate(
            score[start:end], total, cfg.k_min, cfg.k_max)
    assert all(k_uniform[j:j+FC.group_size].sum() == FC.group_size * FC.online_average
               and k_task[j:j+FC.group_size].sum() == FC.group_size * FC.online_average
               for j in range(0,n,FC.group_size))
    arange = torch.arange(cfg.k_max, device=DEVICE)
    inds = torch.as_tensor(image_ids, device=DEVICE, dtype=torch.long)
    preds = {'SelectedUniform': [], 'SelectedTask512': []}
    for rep in range(FC.online_repeats):
        # Same random seeds, sample bank, and selected proposal as the main follow-up experiment.
        g = torch_rng(seed + 8_000_000 + rep)
        z, lw = draw_proposal(model, x_follow, cfg.k_max, g, selected)
        z, lw = z.double(), lw.double()
        for method, counts in [('SelectedUniform',k_uniform), ('SelectedTask512',k_task)]:
            ct = torch.as_tensor(counts, device=DEVICE, dtype=torch.long)
            mask = arange[None, :] < ct[:, None]
            w = torch.softmax(lw.masked_fill(~mask, -float('inf')), dim=1)
            mu = (w[:, :, None] * z).sum(1).index_select(0, inds)
            preds[method].append(mu.cpu().numpy())
    return {method: np.stack(arrays) for method, arrays in preds.items()}

def audit_one_seed(dim, seed, image_ids):
    prefix = FUOUT / f'd{dim}_s{seed}'
    complete = FUOUT / f'd{dim}_s{seed}.complete.json'
    weights_path = FUOUT / f'd{dim}_s{seed}.model.pt'
    regs_path = FUOUT / f'd{dim}_s{seed}.allocation_regressors.joblib'
    if not all(p.exists() for p in (complete, weights_path, regs_path)):
        raise FileNotFoundError(f'Missing existing model/results for d={dim}, seed={seed}. '
                                'Run the original full follow-up experiment first.')
    info = json.loads(complete.read_text())
    # The proposal is frozen from the EXISTING validation-based tuning, not retuned here.
    selected = next((p for p in PROPOSALS if p.name == info['proposal']), None)
    if selected is None:
        raise ValueError(f"Saved proposal {info['proposal']} is not in PROPOSALS")
    model = GaussianVAE(dim, cfg.hidden, cfg.decoder_sigma).to(DEVICE)
    try:
        try:
            state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        regs = joblib.load(regs_path)
        # Frozen online estimates do not depend on the reference size.
        online = online_estimates_from_frozen_allocation(model, selected, regs, seed, image_ids)
        xs = x_follow[image_ids]
        ref_A = reference_checkpoints(model, xs, seed + 31_000_000 + dim * 1000 + 1, selected, N_LIST)
        ref_B = reference_checkpoints(model, xs, seed + 31_000_000 + dim * 1000 + 2, selected, N_LIST)
        rows=[]
        combined={}
        for N in N_LIST:
            A, B = ref_A[N], ref_B[N]
            R = combine_references(A,B)
            combined[N] = R['mu']
            ess_fraction = R['ess']/(2*N)
            row={'dim':dim,'seed':seed,'reference_N_each':N,
                 'n_audit_images':len(xs),'selected_proposal':selected.name,
                 'ref_pair_mse':float(np.mean((A['mu']-B['mu'])**2)),
                 'ref_ess_fraction_median':float(np.median(ess_fraction)),
                 'ref_ess_below_1pct':float(np.mean(ess_fraction < 0.01)),
                 'reference_evaluations':int(len(xs)*2*N)}
            for label, target in [('combined',R['mu']),('A',A['mu']),('B',B['mu'])]:
                uniform_mse=float(np.mean((online['SelectedUniform']-target[None,:,:])**2))
                task_mse=float(np.mean((online['SelectedTask512']-target[None,:,:])**2))
                row[f'uniform_mse_{label}']=uniform_mse
                row[f'task_mse_{label}']=task_mse
                row[f'paired_difference_{label}']=task_mse-uniform_mse
                row[f'reduction_pct_{label}']=100*(1-task_mse/uniform_mse)
            rows.append(row)
        for i, row in enumerate(rows):
            previous_N = N_LIST[i-1] if i > 0 else None
            row['ref_change_from_previous_mse']=(
                float(np.mean((combined[row['reference_N_each']]-combined[previous_N])**2))
                if previous_N is not None else np.nan)
        return pd.DataFrame(rows)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Checkpoint one seed at a time. Re-run this cell after a Colab interruption to resume.
for dim in FC.dims:
    for i in range(SEED_COUNT):
        seed = FC.seed_start + i
        dest = REF_AUDIT_OUT / f'd{dim}_s{seed}.audit.csv'
        if dest.exists():
            print(f'Skipping completed audit d={dim}, seed={seed}')
            continue
        result = audit_one_seed(dim, seed, IMAGE_IDS)
        result.to_csv(dest, index=False)
        print(f'Completed d={dim}, seed={seed}: '
              f'pair MSE {result.ref_pair_mse.iloc[0]:.4g} -> '
              f'{result.ref_pair_mse.iloc[-1]:.4g}; '
              f'full-N reduction {result.reduction_pct_combined.iloc[-1]:+.3f}%')

# Aggregate only the completed seeds; report missing seeds explicitly.
parts = [pd.read_csv(p) for p in sorted(REF_AUDIT_OUT.glob('d*_s*.audit.csv'))]
if not parts:
    raise RuntimeError('No completed audits to summarize')
AUDIT = pd.concat(parts,ignore_index=True)
AUDIT.to_csv(REF_AUDIT_OUT/'all_seed_reference_convergence.csv',index=False)
print('Completed seed counts:',AUDIT[AUDIT.reference_N_each==N_LIST[-1]].groupby('dim').seed.nunique().to_dict())

summary_rows=[]
rng=np.random.default_rng(333)
for (dim,N), g in AUDIT.groupby(['dim','reference_N_each'], sort=True):
    x=g['task_mse_combined'].to_numpy()
    y=g['uniform_mse_combined'].to_numpy()
    d=x-y
    n=len(d)
    diff=float(np.mean(d))
    if n>1:
        half=stats.t.ppf(.975,n-1)*stats.sem(d)
        ci_low,ci_high=diff-half,diff+half
        ids=rng.integers(0,n,size=(REFERENCE_BOOTSTRAPS,n))
        boot_reductions=100*(1-x[ids].mean(1)/y[ids].mean(1))
        pct_lo,pct_hi=np.quantile(boot_reductions,[.025,.975])
        # Exact two-sided sign flip at n<=16; otherwise Monte Carlo.
        if n<=16:
            flips=np.asarray(list(itertools.product((-1.,1.),repeat=n)))
        else:
            flips=rng.choice((-1.,1.),size=(100_000,n))
        p=float(np.mean(np.abs((flips*d).mean(1))>=abs(diff)-1e-15))
    else:
        ci_low=ci_high=pct_lo=pct_hi=p=np.nan
    summary_rows.append({'dim':dim,'reference_N_each':int(N), 'n_seeds':n,
        'mean_reference_pair_mse':g.ref_pair_mse.mean(),
        'mean_reference_change_mse':g.ref_change_from_previous_mse.mean(),
        'median_reference_ess_fraction':g.ref_ess_fraction_median.median(),
        'mean_ref_fraction_below_1pct':g.ref_ess_below_1pct.mean(),
        'mean_task_mse':x.mean(),'mean_uniform_mse':y.mean(),
        'paired_difference':diff,'paired_ci_low':ci_low,'paired_ci_high':ci_high,
        'mse_reduction_pct':100*(1-x.mean()/y.mean()),
        'reduction_bootstrap_ci_low':pct_lo,'reduction_bootstrap_ci_high':pct_hi,
        'paired_signflip_p':p,'seeds_task_lower':int((d<0).sum()),
        'reduction_ref_A_pct':100*(1-g.task_mse_A.mean()/g.uniform_mse_A.mean()),
        'reduction_ref_B_pct':100*(1-g.task_mse_B.mean()/g.uniform_mse_B.mean())})
SUMMARY=pd.DataFrame(summary_rows).sort_values(['dim','reference_N_each'])
SUMMARY.to_csv(REF_AUDIT_OUT/'reference_convergence_summary.csv',index=False)
print('\nREFERENCE CONVERGENCE SUMMARY (exploratory until full 25 seeds / 104 images):')
display(SUMMARY.round(6))

for dim in sorted(SUMMARY.dim.unique()):
    s=SUMMARY[SUMMARY.dim==dim]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(s.reference_N_each,s.mean_reference_pair_mse,'o-',label='Independent A/B disagreement')
    ax.set_xlabel('Draws per independent reference (A and B)')
    ax.set_ylabel('Reference A/B posterior-mean MSE')
    ax.set_title(f'VAE reference convergence, latent d={dim}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(REF_AUDIT_OUT/f'd{dim}_reference_convergence.png',dpi=160)
    plt.show()
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(s.reference_N_each,s.mse_reduction_pct,'o-',label='Task-aware MSE reduction')
    ax.fill_between(s.reference_N_each,s.reduction_bootstrap_ci_low,
                    s.reduction_bootstrap_ci_high,alpha=.2,label='Seed-bootstrap 95% CI')
    ax.axhline(0,linestyle='--',color='grey')
    ax.set_xlabel('Draws per independent reference (A and B)')
    ax.set_ylabel('Task-aware vs uniform MSE reduction (%)')
    ax.set_title(f'Effect stability as reference improves, latent d={dim}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(REF_AUDIT_OUT/f'd{dim}_effect_stability.png',dpi=160)
    plt.show()

print('Results saved at:',REF_AUDIT_OUT)
print('Interpretation: declining A/B disagreement and stable effect estimates are encouraging,')
print('but common proposal bias means these references are not exact posterior ground truth.')
print('Pilot (5 seeds / 16 images) is diagnostic only; do NOT report its p-values in an SOP.')
