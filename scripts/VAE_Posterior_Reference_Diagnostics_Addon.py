"""Independent reference-weight diagnostics and HMC cross-check.

Run with %run -i /content/VAE_Posterior_Reference_Diagnostics_Addon.py
AFTER the original Task_Aware_VAE_Followup_With_Reference_Audit notebook
has defined its VAE/proposal utilities and FUOUT, x_follow, cfg, etc.
No model training is performed by this add-on.
"""
import math, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

if 'FUOUT' not in globals() or 'x_follow' not in globals() or 'GaussianVAE' not in globals():
    raise RuntimeError('Run the VAE follow-up notebook setup/function cells first (FUOUT, x_follow, GaussianVAE are required).')
if 'FC' not in globals() or 'PROPOSALS' not in globals() or 'draw_proposal' not in globals():
    raise RuntimeError('Run the notebook through its follow-up setup before this add-on.')
if FOLLOWUP_MODE != 'full':
    raise RuntimeError('This audit requires the existing FULL follow-up run, not smoke-mode weights.')

CHECK_MODE = globals().get('CHECK_MODE', 'quick')  # 'quick' for smoke, 'full' for evidence
if CHECK_MODE not in ('quick','full'):
    raise ValueError('CHECK_MODE must be quick or full')

# This matches the original reference audit's deterministic pilot image selection.
if 'IMAGE_IDS' in globals() and len(IMAGE_IDS) == 16:
    _IMAGE_IDS = np.asarray(IMAGE_IDS, dtype=np.int64)
else:
    _IMAGE_IDS = np.sort(np.random.default_rng(20261123).choice(len(x_follow),16,replace=False))

REF_REPEATS = 1 if CHECK_MODE == 'quick' else 3
REF_N = 4096 if CHECK_MODE == 'quick' else 131_072
REF_IMAGE_IDS = _IMAGE_IDS[:2] if CHECK_MODE == 'quick' else _IMAGE_IDS
HMC_SEEDS = [27000] if CHECK_MODE == 'quick' else [27000, 27002]
HMC_IMAGE_IDS = _IMAGE_IDS[:1] if CHECK_MODE == 'quick' else _IMAGE_IDS[:3]
HMC_CHAINS = 4
HMC_WARMUP = 80 if CHECK_MODE == 'quick' else 1000
HMC_DRAWS = 120 if CHECK_MODE == 'quick' else 2000
HMC_LF_STEPS = 12
HMC_REF_N = 4096 if CHECK_MODE == 'quick' else 131_072

OUT_DIAG = FUOUT / ('independent_posterior_checks_' + CHECK_MODE)
OUT_DIAG.mkdir(parents=True, exist_ok=True)
(OUT_DIAG / 'configuration.json').write_text(json.dumps({
    'mode':CHECK_MODE,'reference_repeats':REF_REPEATS,'reference_draws_each':REF_N,
    '4d_seed':27000,'reference_image_ids':REF_IMAGE_IDS.tolist(),
    'hmc_seeds':HMC_SEEDS,'hmc_image_ids':HMC_IMAGE_IDS.tolist(),
    'hmc_warmup':HMC_WARMUP,'hmc_samples_per_chain':HMC_DRAWS,
    'hmc_chains':HMC_CHAINS,'hmc_leapfrog_steps':HMC_LF_STEPS,
    'hmc_ref_draws_each':HMC_REF_N},indent=2))
print('Mode:',CHECK_MODE,'\nOutputs:',OUT_DIAG)


def _load_model_and_proposal(dim,seed,dtype=torch.float32):
    complete = FUOUT / f'd{dim}_s{seed}.complete.json'
    weights = FUOUT / f'd{dim}_s{seed}.model.pt'
    if not complete.exists() or not weights.exists():
        raise FileNotFoundError('Missing trained model for '+str((dim,seed))+'. Check that FUOUT points to the completed follow-up run and Drive is mounted if used.')
    saved = json.loads(complete.read_text())
    proposal = next((p for p in PROPOSALS if p.name == saved['proposal']),None)
    if proposal is None:
        raise ValueError('Saved proposal is not defined: '+saved['proposal'])
    model = GaussianVAE(dim,cfg.hidden,cfg.decoder_sigma).to(DEVICE)
    try:
        state=torch.load(weights,map_location=DEVICE,weights_only=True)
    except TypeError:
        state=torch.load(weights,map_location=DEVICE)
    model.load_state_dict(state)
    model=model.to(dtype=dtype).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,proposal


@torch.inference_mode()
def detailed_reference_pass(model,xs,seed,proposal,N,chunk=1024):
    """Independent SNIS reference with per-image ESS, maximum weight share and top-10 mass.

    Uses exactly the notebook's draw_proposal (correct q mixture density).
    All accumulation is in float64 and streaming/logsumexp stabilized.
    """
    g=torch_rng(seed)
    chunks={k:[] for k in ('mu','ess','logz','max_weight_share','top10_weight_share')}
    d=model.latent_dim
    for start in range(0,len(xs),FC.ref_image_batch):
        xb=xs[start:start+FC.ref_image_batch]
        b=len(xb)
        lm=torch.full((b,),-float('inf'),dtype=torch.float64,device=DEVICE)
        sw=torch.zeros(b,dtype=torch.float64,device=DEVICE)
        sw2=torch.zeros(b,dtype=torch.float64,device=DEVICE)
        swz=torch.zeros((b,d),dtype=torch.float64,device=DEVICE)
        top=torch.full((b,10),-float('inf'),dtype=torch.float64,device=DEVICE)
        for first in range(0,N,chunk):
            take=min(chunk,N-first)
            z,lw=draw_proposal(model,xb,take,g,proposal)
            z,lw=z.double(),lw.double()
            new_lm=torch.maximum(lm,lw.max(1).values)
            rescale=torch.exp(lm-new_lm)
            w=torch.exp(lw-new_lm[:,None])
            sw=sw*rescale+w.sum(1)
            sw2=sw2*rescale.square()+(w*w).sum(1)
            swz=swz*rescale[:,None]+(w[:,:,None]*z).sum(1)
            lm=new_lm
            top=torch.topk(torch.cat((top,lw),axis=1),10,dim=1).values
        log_sw=lm+torch.log(sw)
        chunks['mu'].append((swz/sw[:,None]).cpu().numpy())
        chunks['ess'].append((sw.square()/sw2).cpu().numpy())
        chunks['logz'].append((log_sw-math.log(N)).cpu().numpy())
        chunks['max_weight_share'].append(torch.exp(top[:,0]-log_sw).cpu().numpy())
        chunks['top10_weight_share'].append(torch.exp(top-log_sw[:,None]).sum(1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in chunks.items()}


def investigate_4d_seed():
    dim,seed=4,27000
    model,selected=_load_model_and_proposal(dim,seed)
    images=x_follow[REF_IMAGE_IDS]
    lines=[]
    seed_pairs=[]
    if CHECK_MODE == 'full':
        # Reproduce the anomalous 4D 131072-draw pair from the preceding audit.
        original_seed=seed+31_000_000+dim*1000
        seed_pairs.append((-1,original_seed+1,original_seed+2))
    seed_pairs.extend((rep,seed+55_000_001+rep*101,seed+55_000_002+rep*101)
                      for rep in range(REF_REPEATS))
    for rep,sA,sB in seed_pairs:
        A=detailed_reference_pass(model,images,sA,selected,REF_N)
        B=detailed_reference_pass(model,images,sB,selected,REF_N)
        per_img=np.mean((A['mu']-B['mu'])**2,axis=1)
        for j,image_id in enumerate(REF_IMAGE_IDS):
            lines.append({'dim':dim,'seed':seed,'replicate_pair':rep,'image_id':int(image_id),
                'dataset_index':int(fresh_ids[image_id]) if 'fresh_ids' in globals() else np.nan,
                'proposal':selected.name,'N_per_pass':REF_N,
                'reference_stream':('original' if rep==-1 else f'new_{rep+1}'),
                'pair_mse':float(per_img[j]),
                'A_ess_fraction':float(A['ess'][j]/REF_N),'B_ess_fraction':float(B['ess'][j]/REF_N),
                'A_max_weight_share':float(A['max_weight_share'][j]),
                'B_max_weight_share':float(B['max_weight_share'][j]),
                'A_top10_weight_share':float(A['top10_weight_share'][j]),
                'B_top10_weight_share':float(B['top10_weight_share'][j]),
                'A_logZ':float(A['logz'][j]),'B_logZ':float(B['logz'][j]),
                'A_posterior_mean':json.dumps(A['mu'][j].tolist()),
                'B_posterior_mean':json.dumps(B['mu'][j].tolist())})
        print(f'4D seed {seed} stream {"original" if rep==-1 else str(rep+1)+"/"+str(REF_REPEATS)}: A/B MSE {per_img.mean():.6g}; '
              f'max single-image MSE {per_img.max():.6g}')
    df=pd.DataFrame(lines)
    df.to_csv(OUT_DIAG/'4d_seed27000_reference_weight_diagnostics.csv',index=False)
    ranked=df.sort_values('pair_mse',ascending=False)
    display(ranked[['replicate_pair','image_id','pair_mse','A_ess_fraction','B_ess_fraction',
                    'A_max_weight_share','B_max_weight_share','A_top10_weight_share','B_top10_weight_share']].head(15))
    by_pair=df.groupby('replicate_pair').agg(pair_mse=('pair_mse','mean'),
       worst_image_mse=('pair_mse','max'),max_A_weight=('A_max_weight_share','max'),
       max_B_weight=('B_max_weight_share','max'))
    display(by_pair)
    fig,ax=plt.subplots(figsize=(7,4))
    for rep,part in df.groupby('replicate_pair'):
        label='Original audit pair' if rep==-1 else f'New independent pair {rep+1}'
        ax.scatter(part.image_id,part.pair_mse,label=label,alpha=.75)
    ax.set_xlabel('Image ID within x_follow');ax.set_ylabel('Reference A/B MSE (per image)')
    ax.set_title('4D seed 27000: outlying images and replicate variability')
    ax.legend();fig.tight_layout();fig.savefig(OUT_DIAG/'4d_outlier_reference_pairs.png',dpi=160);plt.show()
    return df


def _potential_and_grad(model,x,z):
    """U(z)=-log p(z,x), up to additive constants; exact Gaussian VAE decoder likelihood."""
    z=z.detach().requires_grad_(True)
    with torch.enable_grad():
        rec=model.decoder(z)
        U=.5*z.square().sum(-1)+.5*((x[None,:]-rec)/model.decoder_sigma).square().sum(-1)
        g=torch.autograd.grad(U.sum(),z)[0]
    return U.detach(),g.detach()


def run_fixed_length_hmc(model,x,seed,chains=4,warmup=1000,n_draws=2000,n_leapfrog=12,
                         initial_step=.02,target_accept=.82):
    """Four independent chains, leapfrog, Metropolis correction, dual-averaging step-size adaptation.

    This is a deliberately small independent posterior cross-check, NOT a substitute for NUTS.
    Require R-hat / bulk ESS diagnostics before using samples for scientific conclusions.
    """
    x=x.to(DEVICE,dtype=torch.float64)
    generator=torch.Generator(device=DEVICE.type)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        enc_mu,enc_lv=model.encode(x[None,:])
        starts=enc_mu+torch.exp(.5*enc_lv)*torch.randn((chains,model.latent_dim),
                                                      generator=generator,device=DEVICE,dtype=torch.float64)
    q=starts.detach()
    eps=float(initial_step)
    da_mu=np.log(10*eps);da_logepsbar=np.log(eps);da_H=0.0
    accepted=[];integration_errors=[];draws=[]
    for t in range(1,warmup+n_draws+1):
        p0=torch.randn(q.shape,device=DEVICE,dtype=torch.float64,generator=generator)
        U0,g0=_potential_and_grad(model,x,q)
        q1=q.clone();p=p0-.5*eps*g0
        for l in range(n_leapfrog):
            q1=q1+eps*p
            U1,g1=_potential_and_grad(model,x,q1)
            if l+1<n_leapfrog:p=p-eps*g1
        p=p-.5*eps*g1
        delta=(U0+.5*p0.square().sum(-1))-(U1+.5*p.square().sum(-1))
        delta=torch.nan_to_num(delta,nan=-1e10,posinf=-1e10,neginf=-1e10)
        acceptprob=torch.exp(torch.minimum(delta,torch.zeros_like(delta))).clamp(0,1)
        change=torch.rand((chains,),device=DEVICE,generator=generator)<acceptprob
        q=torch.where(change[:,None],q1,q).detach()
        if t<=warmup:
            # One shared step size, adapted from average accept probability.
            a=float(acceptprob.mean())
            weight=1.0/(t+10)
            da_H=(1-weight)*da_H+weight*(target_accept-a)
            logeps=da_mu-math.sqrt(t)/.05*da_H
            logeps=min(max(logeps,math.log(1e-4)),math.log(.3))
            power=t**(-.75)
            da_logepsbar=power*logeps+(1-power)*da_logepsbar
            eps=math.exp(logeps)
            if t==warmup:
                eps=float(np.clip(math.exp(da_logepsbar),1e-4,.3))
        else:
            accepted.append(change.cpu().numpy())
            integration_errors.append((delta.abs()>100).cpu().numpy())
            draws.append(q.cpu().numpy().copy())
    sample_array=np.stack(draws).transpose(1,0,2)   # (chain,draw,dimension)
    acc=np.stack(accepted).mean(axis=0)
    div=np.stack(integration_errors).sum(axis=0)
    return sample_array,{'step_size':eps,'accept_fraction_by_chain':acc.tolist(),
                         'large_energy_error_by_chain':div.tolist()}


def hmc_diagnostics(draws,info):
    import arviz as az
    # Array API works across ArviZ releases (including >=1.0).
    rhat=np.asarray([float(az.rhat(draws[:,:,j],method='rank')) for j in range(draws.shape[-1])])
    ess=np.asarray([float(az.ess(draws[:,:,j],method='bulk')) for j in range(draws.shape[-1])])
    mu=draws.mean(axis=(0,1))
    # approximate marginal Monte Carlo SE using bulk ESS (not a joint confidence region)
    sd=draws.reshape(-1,draws.shape[-1]).std(axis=0,ddof=1)
    mcse=sd/np.sqrt(np.maximum(ess,1))
    valid=(np.isfinite(rhat).all() and np.isfinite(ess).all()
           and rhat.max()<1.01 and ess.min()>400
           and not any(x>0 for x in info['large_energy_error_by_chain'])
           and np.min(info['accept_fraction_by_chain'])>.5)
    return dict(mu=mu,mcse=mcse,max_rank_rhat=float(rhat.max()),
         min_bulk_ess=float(ess.min()),accept_mean=float(np.mean(info['accept_fraction_by_chain'])),
         large_energy_errors=int(sum(info['large_energy_error_by_chain'])),
         passed_basic_diagnostics=bool(valid))


def investigate_8d_hmc():
    try:
        import arviz
    except ImportError as exc:
        raise ImportError('ArviZ is required. In Colab, run !pip -q install arviz') from exc
    output=[]
    for seed in HMC_SEEDS:
        m_ref,selected=_load_model_and_proposal(8,seed)
        # Dedicated float64 decoder for gradient integration. IS uses original float32 weights.
        m_hmc,_=_load_model_and_proposal(8,seed,dtype=torch.float64)
        xs=x_follow[HMC_IMAGE_IDS]
        # Predictive baselines use the frozen original 104-image allocation groups.
        regs_path=FUOUT/f'd8_s{seed}.allocation_regressors.joblib'
        if regs_path.exists() and 'online_estimates_from_frozen_allocation' in globals():
            import joblib
            regs=joblib.load(regs_path)
            online=online_estimates_from_frozen_allocation(m_ref,selected,regs,seed,HMC_IMAGE_IDS)
        else:
            online=None
        refsA=detailed_reference_pass(m_ref,xs,seed+61_000_001,selected,HMC_REF_N)
        refsB=detailed_reference_pass(m_ref,xs,seed+61_000_002,selected,HMC_REF_N)
        for j,image_id in enumerate(HMC_IMAGE_IDS):
            casefile=OUT_DIAG/f'd8_seed{seed}_img{int(image_id)}.summary.json'
            if casefile.exists():
                r=json.loads(casefile.read_text())
                output.append(r)
                print(f'Skipping previously completed HMC case: seed={seed}, image={int(image_id)}')
                continue
            samp,details=run_fixed_length_hmc(m_hmc,xs[j],seed+70_000_000+int(image_id),
                   chains=HMC_CHAINS,warmup=HMC_WARMUP,n_draws=HMC_DRAWS,
                   n_leapfrog=HMC_LF_STEPS)
            d=hmc_diagnostics(samp,details)
            meanA=refsA['mu'][j];meanB=refsB['mu'][j]
            errorA=float(np.mean((meanA-d['mu'])**2))
            errorB=float(np.mean((meanB-d['mu'])**2))
            r={'dim':8,'seed':seed,'image_id':int(image_id),
               'selected_proposal':selected.name,'draws_per_IS_reference':HMC_REF_N,
               'A_vs_B_mse':float(np.mean((meanA-meanB)**2)),
               'A_vs_HMC_mse':errorA,'B_vs_HMC_mse':errorB,
               'A_ess_fraction':float(refsA['ess'][j]/HMC_REF_N),
               'B_ess_fraction':float(refsB['ess'][j]/HMC_REF_N),
               'A_max_weight_share':float(refsA['max_weight_share'][j]),
               'B_max_weight_share':float(refsB['max_weight_share'][j]),
               'hmc_mean':json.dumps(d['mu'].tolist()),
               'hmc_mcse':json.dumps(d['mcse'].tolist()),
               'IS_A_mean':json.dumps(meanA.tolist()),
               'IS_B_mean':json.dumps(meanB.tolist()),
               **{k:v for k,v in d.items() if k not in ('mu','mcse')},
               'hmc_step_size':details['step_size']}
            if online is not None:
                r['uniform_mse_vs_HMC']=float(np.mean((online['SelectedUniform'][:,j,:]-d['mu'])**2))
                r['task_mse_vs_HMC']=float(np.mean((online['SelectedTask512'][:,j,:]-d['mu'])**2))
            output.append(r)
            np.savez_compressed(OUT_DIAG/f'd8_seed{seed}_img{image_id}_HMC.npz',samples=samp)
            casefile.write_text(json.dumps(r, indent=2))
            print(f'8D seed {seed}, img {int(image_id)}: rhat={d["max_rank_rhat"]:.3f}; '
                  f'ESS_min={d["min_bulk_ess"]:.0f}; HMC diagnostics PASS={d["passed_basic_diagnostics"]}; '
                  f'IS(A)/HMC MSE={errorA:.4g}')
            pd.DataFrame(output).to_csv(OUT_DIAG/'8d_hmc_vs_IS_per_image.csv',index=False)
        del m_ref,m_hmc
        if torch.cuda.is_available():torch.cuda.empty_cache()
    df=pd.DataFrame(output)
    display(df[['seed','image_id','max_rank_rhat','min_bulk_ess','accept_mean',
               'large_energy_errors','passed_basic_diagnostics','A_vs_B_mse','A_vs_HMC_mse','B_vs_HMC_mse']])
    return df

print('TEST 1: repeating 4D seed 27000 independent high-budget reference pairs...')
D4_WEIGHT_AUDIT=investigate_4d_seed()
print('TEST 2: independent 8D posterior cross-check with multiple-chain HMC...')
D8_HMC_AUDIT=investigate_8d_hmc()
print('\nSAVED:',OUT_DIAG)
print('Quick mode is a CODE CHECK ONLY. Full mode is a small diagnostic, not a population-wide confirmation.')
print('HMC comparisons must NOT be interpreted when R-hat/ESS/energy diagnostics fail.')
