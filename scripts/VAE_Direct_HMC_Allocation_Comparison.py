"""Direct, independent HMC posterior-mean comparison for the existing VAE follow-up.

Paste this code as ONE new cell in Task_Aware_VAE_Followup_Colab (1).ipynb
AFTER the existing independent HMC diagnostic cell (which defines
run_fixed_length_hmc, hmc_diagnostics, _load_model_and_proposal) and AFTER
Section 15A (which defines online_estimates_from_frozen_allocation).

The executed Colab cells must point FUOUT at the existing FULL follow-up
checkpoints.  This add-on NEVER refits a VAE or a sampling-difficulty regressor.
"""
import json, hashlib, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from scipy import stats
from IPython.display import display

DIRECT_HMC_MODE = globals().get('DIRECT_HMC_MODE', 'pilot')  # 'pilot' or 'full'
if DIRECT_HMC_MODE not in ('pilot', 'full'):
    raise ValueError('DIRECT_HMC_MODE must be pilot or full')
for required in ['FUOUT', 'FC', 'cfg', 'x_follow', 'DEVICE',
                 'run_fixed_length_hmc', 'hmc_diagnostics',
                 '_load_model_and_proposal',
                 'online_estimates_from_frozen_allocation']:
    if required not in globals():
        raise RuntimeError(f'Missing {required}: run the notebook setup, Section 15A, and the earlier HMC diagnostic cell first.')
if FOLLOWUP_MODE != 'full' or len(x_follow) != 104 or 8 not in FC.dims:
    raise RuntimeError('This add-on requires the completed FULL 8D VAE follow-up, with 104 frozen held-out images.')

# Frozen design; the seeds/images are selected without examining HMC allocation results.
# We exclude the three indices used in the previous 8D HMC diagnostic.
_PREVIOUS_HMC_IDS = np.sort(np.random.default_rng(20261123).choice(len(x_follow), 16, replace=False))[:3]
_rng_design = np.random.default_rng(20261124)
_ALL_SEEDS = sorted(int(v) for v in _rng_design.choice(
    np.arange(FC.seed_start, FC.seed_start + FC.n_seeds), size=8, replace=False))
_ELIGIBLE_IMAGE_IDS = np.array([i for i in range(len(x_follow))
                                if i not in set(_PREVIOUS_HMC_IDS.tolist())], dtype=int)
_ALL_IMAGE_IDS = sorted(int(v) for v in _rng_design.choice(_ELIGIBLE_IMAGE_IDS,
                                                             size=12, replace=False))
# The pilot is a code check only; do not combine its metrics with full-mode results.
HMC_AUDIT_SEEDS = _ALL_SEEDS[:1] if DIRECT_HMC_MODE == 'pilot' else _ALL_SEEDS
HMC_AUDIT_IMAGE_IDS = _ALL_IMAGE_IDS[:2] if DIRECT_HMC_MODE == 'pilot' else _ALL_IMAGE_IDS
HMC_AUDIT_WARMUP = 80 if DIRECT_HMC_MODE == 'pilot' else 1000
HMC_AUDIT_DRAWS = 120 if DIRECT_HMC_MODE == 'pilot' else 2000
HMC_AUDIT_CHAINS = 4
HMC_AUDIT_LF = 12
HMC_REQUIRED_PASS_FRACTION = 0.80  # prespecified coverage threshold for reporting aggregate comparisons
HMC_REQUIRED_MIN_PASS_PER_SEED = 6 # of the twelve images per seed in full mode

_AUDIT_CONFIG = {
    'mode': DIRECT_HMC_MODE, 'seeds': HMC_AUDIT_SEEDS,
    'image_ids': HMC_AUDIT_IMAGE_IDS,
    'excluded_previously_inspected_image_ids': _PREVIOUS_HMC_IDS.tolist(),
    'image_selection_seed': 20261124,
    'chains': HMC_AUDIT_CHAINS, 'warmup': HMC_AUDIT_WARMUP,
    'samples_per_chain': HMC_AUDIT_DRAWS, 'leapfrog_steps': HMC_AUDIT_LF,
    'online_repeats': FC.online_repeats, 'online_average_budget': FC.online_average,
    'original_group_size': FC.group_size,
    'max_draws_per_image_in_online_bank': cfg.k_max,
    'pass_fraction_required': HMC_REQUIRED_PASS_FRACTION,
    'min_pass_per_seed': HMC_REQUIRED_MIN_PASS_PER_SEED,
    'interpretation': 'exploratory subset chosen after inspecting earlier follow-up results'
}
_audit_hash = hashlib.sha256(json.dumps(_AUDIT_CONFIG,sort_keys=True).encode()).hexdigest()[:10]
HMC_AUDIT_DIR = FUOUT / f'direct_hmc_allocation_{DIRECT_HMC_MODE}_{_audit_hash}'
HMC_AUDIT_DIR.mkdir(parents=True,exist_ok=True)
_config_path = HMC_AUDIT_DIR / 'frozen_design.json'
if _config_path.exists() and json.loads(_config_path.read_text()) != _AUDIT_CONFIG:
    raise RuntimeError('Frozen configuration changed for an existing run directory.')
_config_path.write_text(json.dumps(_AUDIT_CONFIG,indent=2))
print('DIRECT HMC allocation audit:', DIRECT_HMC_MODE)
print('Output:', HMC_AUDIT_DIR)
print('8D seeds:', HMC_AUDIT_SEEDS)
print('Image IDs within x_follow:', HMC_AUDIT_IMAGE_IDS)
print('Online: exact original 52-image budgets,', FC.online_repeats, 'matched repeats')
if DIRECT_HMC_MODE == 'pilot':
    print('PILOT ONLY: abbreviated HMC diagnostics are expected to fail. No scientific inference.')


def _load_saved_online(seed, model, proposal):
    """Use the original random-number bank and 104-image score/group allocations.

    Original same-seed random draws for Task and Uniform make the comparison paired.
    """
    path = HMC_AUDIT_DIR / f'd8_seed{seed}_frozen_online.npz'
    if path.exists():
        with np.load(path) as z:
            return {k:z[k] for k in ('SelectedUniform','SelectedTask512')}
    regfile = FUOUT / f'd8_s{seed}.allocation_regressors.joblib'
    if not regfile.exists():
        raise FileNotFoundError(f'Missing frozen allocation model: {regfile}')
    regs=joblib.load(regfile)
    if 'Task512' not in regs:
        raise KeyError('The saved allocation regressors do not contain Task512')
    preds=online_estimates_from_frozen_allocation(model,proposal,regs,seed,HMC_AUDIT_IMAGE_IDS)
    for k in ('SelectedUniform','SelectedTask512'):
        assert preds[k].shape == (FC.online_repeats,len(HMC_AUDIT_IMAGE_IDS),8), preds[k].shape
        assert np.isfinite(preds[k]).all()
    np.savez_compressed(path,**preds)
    return preds


def _hmc_per_image(seed, image_id, model_hmc):
    outfile=HMC_AUDIT_DIR / f'd8_seed{seed}_img{image_id}_hmc.json'
    if outfile.exists():
        return json.loads(outfile.read_text())
    x=x_follow[image_id]
    hmc_seed=int(seed + 81_000_000 + image_id)
    samples,info=run_fixed_length_hmc(
        model_hmc,x,hmc_seed,chains=HMC_AUDIT_CHAINS,
        warmup=HMC_AUDIT_WARMUP,n_draws=HMC_AUDIT_DRAWS,
        n_leapfrog=HMC_AUDIT_LF)
    diag=hmc_diagnostics(samples,info)
    if not np.isfinite(samples).all():
        raise FloatingPointError(f'Non-finite HMC draws seed={seed} image={image_id}')
    # Two disjoint chain groups provide a useful, non-independent-of-run sensitivity check.
    halfA=samples[:2].mean(axis=(0,1))
    halfB=samples[2:].mean(axis=(0,1))
    record={
       'seed':int(seed),'image_id':int(image_id),'hmc_rng_seed':hmc_seed,
       'mean':diag['mu'].tolist(),'chain12_mean':halfA.tolist(),
       'chain34_mean':halfB.tolist(),'mcse':diag['mcse'].tolist(),
       'max_rank_rhat':diag['max_rank_rhat'],'min_bulk_ess':diag['min_bulk_ess'],
       'accept_mean':diag['accept_mean'],'large_energy_errors':diag['large_energy_errors'],
       'passed_basic_diagnostics':bool(diag['passed_basic_diagnostics']),
       'hmc_step_size':float(info['step_size']),
       'half_group_mean_disagreement_mse':float(np.mean((halfA-halfB)**2))
    }
    outfile.write_text(json.dumps(record,indent=2))
    return record


def _paired_inference(seed_rows, metric_u, metric_t, label):
    p=seed_rows[[metric_u,metric_t]].dropna()
    n=len(p)
    if n<2:return {'reference':label,'n_seeds':n,'warning':'Too few valid seeds'}
    u=p[metric_u].to_numpy();t=p[metric_t].to_numpy();delta=t-u
    avg=float(delta.mean())
    lo,hi=stats.t.interval(.95,df=n-1,loc=avg,scale=stats.sem(delta)) if np.std(delta,ddof=1)>0 else (avg,avg)
    # Exact two-sided sign-flip at the SEED level; 8 seeds => 256 assignments.
    flips=np.array(list(itertools.product([-1.,1.],repeat=n)))
    flipstats=(flips*delta).mean(axis=1)
    pvalue=float((np.abs(flipstats)>=abs(avg)-1e-15).mean())
    rng=np.random.default_rng(20261125+n)
    ids=rng.integers(0,n,size=(10000,n))
    bperc=100*(1-t[ids].mean(axis=1)/u[ids].mean(axis=1))
    return {'reference':label,'n_seeds':n,
        'mean_uniform_mse':float(u.mean()),'mean_task_mse':float(t.mean()),
        'paired_difference_task_minus_uniform':avg,
        'paired_t95ci_low':float(lo),'paired_t95ci_high':float(hi),
        'relative_reduction_percent':float(100*(1-t.mean()/u.mean())),
        'reduction_bootstrap_ci_low':float(np.quantile(bperc,.025)),
        'reduction_bootstrap_ci_high':float(np.quantile(bperc,.975)),
        'exact_two_sided_signflip_p':pvalue,'seeds_task_lower':int((delta<0).sum())}


_all=[]
for seed in HMC_AUDIT_SEEDS:
    print(f'\n8D seed {seed}: loading frozen model, selected proposal, and online allocations')
    base=FUOUT/f'd8_s{seed}'
    for suffix in ('.complete.json','.model.pt','.allocation_regressors.joblib','.reference.npz'):
        if not Path(str(base)+suffix).exists():
            raise FileNotFoundError(f'Missing saved full follow-up artifact: {base}{suffix}')
    model32,proposal=_load_model_and_proposal(8,seed,dtype=torch.float32)
    model64,_=_load_model_and_proposal(8,seed,dtype=torch.float64)
    estimates=_load_saved_online(seed,model32,proposal)
    with np.load(str(base)+'.reference.npz') as ref0:
        original_reference=np.asarray(ref0['mu'],dtype=np.float64)
    for local_i,image_id in enumerate(HMC_AUDIT_IMAGE_IDS):
        hmc=_hmc_per_image(seed,image_id,model64)
        pred_u=estimates['SelectedUniform'][:,local_i,:]
        pred_t=estimates['SelectedTask512'][:,local_i,:]
        assert pred_u.shape == pred_t.shape == (FC.online_repeats,8)
        refmeans={'HMC':np.asarray(hmc['mean'],dtype=float),
                  'HMC_chains12':np.asarray(hmc['chain12_mean'],dtype=float),
                  'HMC_chains34':np.asarray(hmc['chain34_mean'],dtype=float),
                  'Original_IS':original_reference[image_id]}
        result={k:hmc[k] for k in ('seed','image_id','max_rank_rhat','min_bulk_ess',
             'accept_mean','large_energy_errors','passed_basic_diagnostics',
             'half_group_mean_disagreement_mse')}
        result['original_selected_proposal']=proposal.name
        result['original_dataset_index']=int(fresh_ids[image_id])
        result['hmc_to_original_is_mse']=float(np.mean((refmeans['HMC']-refmeans['Original_IS'])**2))
        for key,ref in refmeans.items():
            result[f'{key}_uniform_mse']=float(np.mean((pred_u-ref[None,:])**2))
            result[f'{key}_task_mse']=float(np.mean((pred_t-ref[None,:])**2))
            result[f'{key}_paired_difference']=result[f'{key}_task_mse']-result[f'{key}_uniform_mse']
        # Approximate worst-case sensitivity to marginal HMC MCSE, not a formal CI.
        mcse=np.asarray(hmc['mcse'])
        diff=np.mean(pred_t-pred_u,axis=0)
        result['hmc_mcse_gap_sensitivity_upper']=float((2/8)*np.sum(np.abs(diff)*mcse))
        _all.append(result)
        pd.DataFrame(_all).to_csv(HMC_AUDIT_DIR/'hmc_vs_allocation_per_image.csv',index=False)
        print(f'  image {image_id:3d}: pass={hmc["passed_basic_diagnostics"]}, '
              f'Rhat={hmc["max_rank_rhat"]:.4f}, bulkESS={hmc["min_bulk_ess"]:.0f}, '
              f'Task−Uniform={result["HMC_paired_difference"]:+.5g}')
    del model32,model64
    if torch.cuda.is_available(): torch.cuda.empty_cache()

case_df=pd.DataFrame(_all)
case_df.to_csv(HMC_AUDIT_DIR/'hmc_vs_allocation_per_image.csv',index=False)
pass_df=case_df[case_df.passed_basic_diagnostics].copy()
coverage=case_df.groupby('seed',as_index=False).agg(
    attempted=('image_id','count'),passed=('passed_basic_diagnostics','sum'),
    max_rhat=('max_rank_rhat','max'),min_ess=('min_bulk_ess','min'))
coverage['pass_fraction']=coverage.passed/coverage.attempted
coverage.to_csv(HMC_AUDIT_DIR/'hmc_coverage_by_seed.csv',index=False)
print('\nHMC diagnostic coverage:')
display(coverage)

if DIRECT_HMC_MODE=='pilot':
    print('\nPILOT ONLY: short HMC runs are for code validation; do NOT interpret effect estimates.')
else:
    # Unsuccessful diagnostics are always reported; never count these as validated HMC references.
    if len(pass_df)==0:
        print('NO HMC CASE PASSED. Investigate the sampler before making any effect claims.')
    else:
        seedwise=pass_df.groupby('seed',as_index=False).agg(
            n_hmc_passed=('image_id','count'),
            **{f'{ref}_{suffix}':(f'{ref}_{suffix}','mean')
               for ref in ('HMC','HMC_chains12','HMC_chains34','Original_IS')
               for suffix in ('uniform_mse','task_mse')})
        seedwise.to_csv(HMC_AUDIT_DIR/'seed_level_hmc_effects.csv',index=False)
        eligible_seeds=seedwise[seedwise.n_hmc_passed >= HMC_REQUIRED_MIN_PASS_PER_SEED].copy()
        findings=[]
        for ref in ('HMC','HMC_chains12','HMC_chains34','Original_IS'):
            findings.append(_paired_inference(
                eligible_seeds,f'{ref}_uniform_mse',f'{ref}_task_mse',ref))
        summary=pd.DataFrame(findings)
        summary.to_csv(HMC_AUDIT_DIR/'seed_level_hmc_comparisons.csv',index=False)
        total_pass_fraction=len(pass_df)/len(case_df)
        sufficient=(total_pass_fraction>=HMC_REQUIRED_PASS_FRACTION and
                    len(eligible_seeds)>=6)
        print('\nHMC reference coverage:',f'{len(pass_df)}/{len(case_df)} ({total_pass_fraction:.1%})')
        print('Seeds with >=6 passing HMC images:',len(eligible_seeds))
        print('Meets prespecified coverage rule:',sufficient)
        print('\nPer-seed comparison (only diagnostically passing HMC references):')
        display(seedwise)
        print('\nEffect estimates and seed-level uncertainty:')
        display(summary)
        if not sufficient:
            print('WARNING: insufficient HMC coverage for an overall positive claim. Report this limitation.')
        else:
            print('HMC comparison is an exploratory subset, NOT a new preregistered confirmation.')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        a=seedwise['HMC_task_mse'].to_numpy();b=seedwise['HMC_uniform_mse'].to_numpy()
        ax.scatter(b,a)
        lo=min(a.min(),b.min());hi=max(a.max(),b.max())
        ax.plot([lo,hi],[lo,hi],'--',label='Equal MSE')
        ax.set(xlabel='Uniform MSE vs HMC',ylabel='Task-aware MSE vs HMC',
               title='8D VAEs: task-aware vs uniform against HMC')
        ax.legend();fig.tight_layout();fig.savefig(HMC_AUDIT_DIR/'hmc_paired_seed_mse.png',dpi=170)
        plt.show();plt.close(fig)
print('\nSaved output:',HMC_AUDIT_DIR)
