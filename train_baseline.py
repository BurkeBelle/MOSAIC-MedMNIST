#!/usr/bin/env python3
"""PEFT Baseline Experiments (LoRA / VPT x Independent / Joint)."""
import os, sys, json, time, copy, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from typing import Dict, List
from config.datasets import ALL_DATASETS, DATASETS_2D, DATASETS_3D, get_num_classes_list, get_dataset_config, get_expert_id
from dataloader.medmnist_loader import create_all_dataloaders, create_dataloader
from model.baseline_model import create_baseline_model, BaselineModel, BaselineTeacher
from engine.evaluator import evaluate_single_dataset, evaluate_all_datasets, print_evaluation_results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='joint', choices=['joint','independent'])
    p.add_argument('--adapter', default='lora', choices=['lora','vpt'])
    p.add_argument('--run_all', action='store_true')
    p.add_argument('--data_root', default='./data_224')
    p.add_argument('--pretrained', default='./vit_base_patch16_224.npz')
    p.add_argument('--output_dir', default='./output_baseline')
    p.add_argument('--lora_rank', type=int, default=8)
    p.add_argument('--lora_alpha', type=float, default=16.0)
    p.add_argument('--num_prompts', type=int, default=10)
    p.add_argument('--num_rounds', type=int, default=72)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_rounds', type=int, default=5)
    p.add_argument('--consist_weight', type=float, default=0.1)
    p.add_argument('--ema_momentum', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--early_stopping_patience', type=int, default=5)
    p.add_argument('--ind_epochs', type=int, default=50)
    p.add_argument('--ind_lr', type=float, default=1e-3)
    p.add_argument('--ind_patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', nargs='+', type=int, default=[42,123,456])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def train_independent_single(ds_name, adapter_mode, args, device):
    cfg = get_dataset_config(ds_name)
    kw = {'lora_rank':args.lora_rank,'lora_alpha':args.lora_alpha} if adapter_mode=='lora' else {'num_prompts':args.num_prompts}
    model = BaselineModel(num_classes_list=[cfg.num_classes], adapter_mode=adapter_mode, **kw)
    if args.pretrained and os.path.exists(args.pretrained):
        from model.baseline_model import _load_npz_weights
        _load_npz_weights(model, args.pretrained)
    model.freeze_backbone(); model.to(device)
    tr_loader = create_dataloader(ds_name,'train',args.data_root,args.batch_size,args.num_workers,False)
    va_loader = create_dataloader(ds_name,'val',args.data_root,args.batch_size,args.num_workers,False)
    te_loader = create_dataloader(ds_name,'test',args.data_root,args.batch_size,args.num_workers,False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.ind_lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ind_epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss() if cfg.task_type=='multi-label' else nn.CrossEntropyLoss()
    best_acc, best_st, pat = 0.0, None, 0
    for ep in range(args.ind_epochs):
        model.train(); tot=0; nb=0
        for imgs, labs in tr_loader:
            imgs=imgs.to(device); labs=labs.to(device)
            labs = labs.float() if cfg.task_type=='multi-label' else labs.long().squeeze()
            _,logits=model(imgs,task_id=0); loss=loss_fn(logits,labs)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params,args.max_grad_norm); opt.step()
            tot+=loss.item(); nb+=1
        sch.step()
        vm = evaluate_single_dataset(model,va_loader,ds_name,0,device)
        if vm['acc']>best_acc: best_acc=vm['acc']; best_st=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=args.ind_patience: break
    if best_st: model.load_state_dict(best_st)
    tm = evaluate_single_dataset(model,te_loader,ds_name,0,device)
    print(f"  {ds_name}: ACC={tm['acc']*100:.2f}%, AUC={tm['auc']:.4f}")
    return tm

def run_independent(adapter_mode, args, device):
    name = f"{adapter_mode}_independent_seed{args.seed}"
    print(f"\n{'#'*60}\n  {name}\n{'#'*60}")
    results = {}
    for ds in ALL_DATASETS:
        print(f"\n  Training {ds}...")
        results[ds] = train_independent_single(ds, adapter_mode, args, device)
    accs=[r['acc'] for r in results.values()]; aucs=[r['auc'] for r in results.values()]
    results['mean_acc']=float(np.mean(accs)); results['mean_auc']=float(np.mean(aucs))
    results['mean_acc_2d']=float(np.mean([results[d]['acc'] for d in DATASETS_2D]))
    results['mean_acc_3d']=float(np.mean([results[d]['acc'] for d in DATASETS_3D]))
    print_evaluation_results(results, ALL_DATASETS, title=name)
    sd=os.path.join(args.output_dir,name); os.makedirs(sd,exist_ok=True)
    with open(os.path.join(sd,'results.json'),'w') as f: json.dump(results,f,indent=2,default=float)
    return results

def run_joint(adapter_mode, args, device):
    name = f"{adapter_mode}_joint_seed{args.seed}"
    print(f"\n{'#'*60}\n  {name}\n{'#'*60}")
    ds_list=ALL_DATASETS; ncl=get_num_classes_list(ds_list)
    kw = {'lora_rank':args.lora_rank,'lora_alpha':args.lora_alpha} if adapter_mode=='lora' else {'num_prompts':args.num_prompts}
    student,teacher = create_baseline_model(ncl, adapter_mode, args.pretrained, True, **kw)
    student.to(device); teacher.to(device)
    dls = create_all_dataloaders(ds_list, args.data_root, args.batch_size, args.num_workers, True)
    params=[p for p in student.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR,SequentialLR
    w=LinearLR(opt,start_factor=0.01,end_factor=1.0,total_iters=args.warmup_rounds)
    c=CosineAnnealingLR(opt,T_max=args.num_rounds-args.warmup_rounds,eta_min=1e-6)
    sch=SequentialLR(opt,[w,c],milestones=[args.warmup_rounds])
    ce=nn.CrossEntropyLoss(); bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss()
    best_acc,best_st,no_imp=0.0,None,0
    sd=os.path.join(args.output_dir,name); os.makedirs(sd,exist_ok=True)
    for ri in range(args.num_rounds):
        student.train(); teacher.eval(); rl=0.0
        for tid,ds in enumerate(ds_list):
            cfg=get_dataset_config(ds); eid=None; lfn=bce if cfg.task_type=='multi-label' else ce
            el,n=0.0,0
            for batch in dls[ds]['train']:
                imgs,labs=batch
                if isinstance(imgs,(tuple,list)): is_,it_=imgs[0].to(device),imgs[1].to(device)
                else: is_=it_=imgs.to(device)
                labs=labs.to(device)
                labs=labs.float() if cfg.task_type=='multi-label' else labs.long().squeeze()
                fs,logits=student(is_,task_id=tid,expert_id=eid)
                with torch.no_grad(): ft=teacher(it_,task_id=tid,return_features=True,expert_id=eid)
                loss=lfn(logits,labs)+args.consist_weight*mse(fs,ft.detach())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params,args.max_grad_norm); opt.step()
                teacher.ema_update(student,args.ema_momentum,is_3d=cfg.is_3d,expert_id=eid)
                el+=loss.item(); n+=1
            rl+=el/max(n,1)
        sch.step()
        vr=evaluate_all_datasets(student,dls,ds_list,device,'val')
        ca=vr.get('mean_acc',0)
        imp=ca>best_acc
        if imp: best_acc=ca; best_st=copy.deepcopy(student.state_dict()); no_imp=0
        else: no_imp+=1
        m='*' if imp else ''
        print(f"  Round {ri+1:3d} | val_acc={ca*100:.2f}% | best={best_acc*100:.2f}% {m}")
        if no_imp>=args.early_stopping_patience: print(f"  Early stop at round {ri+1}"); break
    if best_st: student.load_state_dict(best_st)
    tr=evaluate_all_datasets(student,dls,ds_list,device,'test')
    print_evaluation_results(tr,ds_list,title=name)
    with open(os.path.join(sd,'results.json'),'w') as f: json.dump(tr,f,indent=2,default=float)
    return tr

def main():
    args=parse_args(); device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.run_all:
        all_r={}
        for seed in args.seeds:
            args.seed=seed; set_seed(seed)
            for ad in ['lora','vpt']:
                for mo in ['independent','joint']:
                    k=f"{ad}_{mo}_seed{seed}"; print(f"\n{'='*60}\n  {k}\n{'='*60}")
                    r = run_independent(ad,args,device) if mo=='independent' else run_joint(ad,args,device)
                    all_r[k]={'mean_acc':r.get('mean_acc',0),'mean_auc':r.get('mean_auc',0)}
        print(f"\n{'='*70}\n  Summary (mean +/- std)\n{'='*70}")
        for ad in ['lora','vpt']:
            for mo in ['independent','joint']:
                ac=[all_r[f"{ad}_{mo}_seed{s}"]['mean_acc'] for s in args.seeds]
                print(f"  {ad.upper():4s} {mo:12s}: ACC={np.mean(ac)*100:.2f} +/- {np.std(ac)*100:.2f}%")
        os.makedirs(args.output_dir,exist_ok=True)
        with open(os.path.join(args.output_dir,'summary.json'),'w') as f: json.dump(all_r,f,indent=2,default=float)
    else:
        set_seed(args.seed)
        if args.mode=='independent': run_independent(args.adapter,args,device)
        else: run_joint(args.adapter,args,device)

if __name__=='__main__': main()
