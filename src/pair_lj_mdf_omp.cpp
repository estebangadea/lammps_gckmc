
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
 Contributing author: Jibao Lu (U of Utah)
 ------------------------------------------------------------------------- */

#include <math.h>
#include "pair_lj_mdf_omp.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"

#include "suffix.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJMDFOMP::PairLJMDFOMP(LAMMPS *lmp) :
PairLJMDF(lmp), ThrOMP(lmp, THR_PAIR)
{
    suffix_flag |= Suffix::OMP; // first defined in pair.h
    respa_enable = 0;           // first defined in pair.h
    
}

/* ---------------------------------------------------------------------- */

void PairLJMDFOMP::compute(int eflag, int vflag)
{
    if (eflag || vflag) {
        ev_setup(eflag,vflag);
    } else evflag = vflag_fdotr = 0;
    
    const int nall = atom->nlocal + atom->nghost;
    const int nthreads = comm->nthreads;
    const int inum = list->inum;
    
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(eflag,vflag)
#endif
    {
        int ifrom, ito, tid;
        
        loop_setup_thr(ifrom, ito, tid, inum, nthreads);
        ThrData *thr = fix->get_thr(tid);
        thr->timer(Timer::START);
        ev_setup_thr(eflag, vflag, nall, eatom, vatom, thr);
        
        if (evflag) {
            if (eflag) {
                if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
                else eval<1,1,0>(ifrom, ito, thr);
            } else {
                if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
                else eval<1,0,0>(ifrom, ito, thr);
            }
        } else {
            if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
            else eval<0,0,0>(ifrom, ito, thr);
        }
        thr->timer(Timer::PAIR);
        reduce_thr(this, eflag, vflag, thr);
    } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJMDFOMP::eval(int iifrom, int iito, ThrData * const thr)
{
    const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    const int * _noalias const type = atom->type;
    const double * _noalias const special_lj = force->special_lj;
    const int * _noalias const ilist = list->ilist;
    const int * _noalias const numneigh = list->numneigh;
    const int * const * const firstneigh = list->firstneigh;
    
    double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
    double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;
    
    double rr, d, dd, tt, dt, dp, philj;    // added by Jibao, got from pair_lj_mdf.cpp
    
    const int nlocal = atom->nlocal;
    int j,jj,jnum,jtype;
    
    evdwl = 0.0;
    
    // loop over neighbors of my atoms
    
    for (int ii = iifrom; ii < iito; ++ii) {
        const int i = ilist[ii];
        const int itype = type[i];
        const int    * _noalias const jlist = firstneigh[i];
        const double * _noalias const cutsqi = cutsq[itype];
        //const double * _noalias const offseti = offset[itype];
        const double * _noalias const lj1i = lj1[itype];
        const double * _noalias const lj2i = lj2[itype];
        const double * _noalias const lj3i = lj3[itype];
        const double * _noalias const lj4i = lj4[itype];
        
        const double * _noalias const cuti = cut[itype]; // added by Jibao
        const double * _noalias const cut_inneri = cut_inner[itype]; // added by Jibao
        
        const double * _noalias const cut_inner_sqi = cut_inner_sq[itype]; // added by Jibao
        
        xtmp = x[i].x;
        ytmp = x[i].y;
        ztmp = x[i].z;
        jnum = numneigh[i];
        fxtmp=fytmp=fztmp=0.0;
        
        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;
            
            delx = xtmp - x[j].x;
            dely = ytmp - x[j].y;
            delz = ztmp - x[j].z;
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];
            
            if (rsq < cutsqi[jtype]) {
                r2inv = 1.0/rsq;
                r6inv = r2inv*r2inv*r2inv;
                forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
                
                // added by Jibao, from pair_lj_mdf.cpp
                if (rsq > cut_inner_sqi[jtype]) {
                    philj = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]);
                    
                    rr = sqrt(rsq);
                    dp = (cuti[jtype] - cut_inneri[jtype]);
                    d = (rr-cut_inneri[jtype]) / dp;
                    dd = 1.-d;
                    // taperig function - mdf style
                    tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
                    // minus the derivative of the tapering function
                    dt = 30.* d*d * dd*dd * rr / dp;
                    
                    forcelj = forcelj*tt + philj*dt;
                } else {
                    tt = 1;
                }   // added by Jibao, from pair_lj_mdf.cpp
                
                
                fpair = factor_lj*forcelj*r2inv;
                
                fxtmp += delx*fpair;
                fytmp += dely*fpair;
                fztmp += delz*fpair;
                if (NEWTON_PAIR || j < nlocal) {
                    f[j].x -= delx*fpair;
                    f[j].y -= dely*fpair;
                    f[j].z -= delz*fpair;
                }
                
                if (EFLAG) {
                    evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]);
                    
                    if (rsq > cut_inner_sqi[jtype]) evdwl *= tt;
                    
                    evdwl *= factor_lj;
                }
                
                if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                         evdwl,0.0,fpair,delx,dely,delz,thr);
            }
        }
        f[i].x += fxtmp;
        f[i].y += fytmp;
        f[i].z += fztmp;
    }
    
    // there is a line: if (vflag_fdotr) virial_fdotr_compute(); that is gone for the omp version of pair styles. I still don't know why!
}

/* ---------------------------------------------------------------------- */

double PairLJMDFOMP::memory_usage()
{
    double bytes = memory_usage_thr();
    bytes += PairLJMDF::memory_usage();
    
    return bytes;
}