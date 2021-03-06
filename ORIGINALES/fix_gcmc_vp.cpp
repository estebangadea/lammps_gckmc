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
   Contributing author: Paul Crozier, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_gcmc_vp.h"
#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_hybrid.h"
#include "molecule.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "compute.h"
#include "group.h"
#include "domain.h"
#include "region.h"
#include "random_park.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "thermo.h"
#include "output.h"
#include "neighbor.h"
#include <iostream>
#include "pair_hybrid_overlay.h"    // added by Jibao
#include "pair_sw.h"        // added by Jibao
#include "pair_sw0.h"        // added by Jibao
#include "pair_hybrid.h"        // added by Jibao


using namespace std;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};

/* ---------------------------------------------------------------------- */

FixGCMCVp::FixGCMCVp(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR,"Illegal fix gcmc command");

  if (atom->molecular == 2)
    error->all(FLERR,"Fix gcmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  //size_vector = 8; // commented out by Jibao
    size_vector = 9;    // added by Jibao according to Matias' 2012 lammps version, to output energyout
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args

  nevery = force->inumeric(FLERR,arg[3]);
  nexchanges = force->inumeric(FLERR,arg[4]);
  nmcmoves = force->inumeric(FLERR,arg[5]);
  ngcmc_type = force->inumeric(FLERR,arg[6]);
  seed = force->inumeric(FLERR,arg[7]);
  reservoir_temperature = force->numeric(FLERR,arg[8]);
  chemical_potential = force->numeric(FLERR,arg[9]);
  displace = force->numeric(FLERR,arg[10]);

  if (nexchanges < 0) error->all(FLERR,"Illegal fix gcmc command");
  if (nmcmoves < 0) error->all(FLERR,"Illegal fix gcmc command");
  if (seed <= 0) error->all(FLERR,"Illegal fix gcmc command");
  if (reservoir_temperature < 0.0)
    error->all(FLERR,"Illegal fix gcmc command");
  if (displace < 0.0) error->all(FLERR,"Illegal fix gcmc command");
    
    //molflag = 0; // variable in 2012 verion // Jibao
    pairflag = 0; // added by Jibao. from Matias
    //pressflag=0;    // added by Jibao. from Matias
    regionflag=0;   // added by Jibao. from Matias

  // read options from end of input line

  options(narg-11,&arg[11]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // error checks on region and its extent being inside simulation box

  region_xlo = region_xhi = region_ylo = region_yhi =
    region_zlo = region_zhi = 0.0;
  if (regionflag) {
    if (domain->regions[iregion]->bboxflag == 0)
      error->all(FLERR,"Fix gcmc region does not support a bounding box");
    if (domain->regions[iregion]->dynamic_check())
      error->all(FLERR,"Fix gcmc region cannot be dynamic");

    region_xlo = domain->regions[iregion]->extent_xlo;
    region_xhi = domain->regions[iregion]->extent_xhi;
    region_ylo = domain->regions[iregion]->extent_ylo;
    region_yhi = domain->regions[iregion]->extent_yhi;
    region_zlo = domain->regions[iregion]->extent_zlo;
    region_zhi = domain->regions[iregion]->extent_zhi;

    if (region_xlo < domain->boxlo[0] || region_xhi > domain->boxhi[0] ||
        region_ylo < domain->boxlo[1] || region_yhi > domain->boxhi[1] ||
        region_zlo < domain->boxlo[2] || region_zhi > domain->boxhi[2])
      error->all(FLERR,"Fix gcmc region extends outside simulation box");

    // estimate region volume using MC trials

    double coord[3];
    int inside = 0;
    int attempts = 10000000;
    for (int i = 0; i < attempts; i++) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      if (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) != 0)
        inside++;
    }

    double max_region_volume = (region_xhi - region_xlo)*
     (region_yhi - region_ylo)*(region_zhi - region_zlo);

    region_volume = max_region_volume*static_cast<double> (inside)/
     static_cast<double> (attempts);
  }

  // error check and further setup for mode = MOLECULE

  if (mode == MOLECULE) {
    if (onemols[imol]->xflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have coordinates");
    if (onemols[imol]->typeflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have atom types");
    if (ngcmc_type != 0)
      error->all(FLERR,"Atom type must be zero in fix gcmc mol command");
    if (onemols[imol]->qflag == 1 && atom->q == NULL)
      error->all(FLERR,"Fix gcmc molecule has charges, but atom style does not");

    if (atom->molecular == 2 && onemols != atom->avec->onemols)
      error->all(FLERR,"Fix gcmc molecule template ID must be same "
                 "as atom_style template ID");
    onemols[imol]->check_attributes(0);
  }

  if (charge_flag && atom->q == NULL)
    error->all(FLERR,"Fix gcmc atom has charge, but atom style does not");

  if (shakeflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix gcmc shake and not molecule");

  // setup of coords and imageflags array

  if (mode == ATOM) natoms_per_molecule = 1;
  else natoms_per_molecule = onemols[imol]->natoms;
  memory->create(coords,natoms_per_molecule,3,"gcmc:coords");
  memory->create(imageflags,natoms_per_molecule,"gcmc:imageflags");
  memory->create(atom_coord,natoms_per_molecule,3,"gcmc:atom_coord");

  // compute the number of MC cycles that occur nevery timesteps

  ncycles = nexchanges + nmcmoves;

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  ntranslation_attempts = 0.0;
  ntranslation_successes = 0.0;
  nrotation_attempts = 0.0;
  nrotation_successes = 0.0;
  ndeletion_attempts = 0.0;
  ndeletion_successes = 0.0;
  ninsertion_attempts = 0.0;
  ninsertion_successes = 0.0;
    
    energyout=0.0;  // Matias

  gcmc_nmax = 0;
  local_gas_list = NULL;
    
    /*
    if(pairflag==1){                 //pairflag stw definido en el lammps
        pairsw = new PairSW(lmp);
        char *a[6];
        a[0] ="*";
        a[1] ="*";
        a[2] ="NaCl.sw";
        a[3]= "mW";
        a[4]="Na";
        a[5]="Cl";
        pairsw->coeff(6,a);
        //printf("Este es el cut max %f/n",PairSW->cutmax);
    } // from Matias, added by Jibao
    */
    
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixGCMCVp::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix gcmc command");

  // defaults

  mode = ATOM;
  max_rotation_angle = 10*MY_PI/180;
  regionflag = 0;
  iregion = -1;
  region_volume = 0;
  max_region_attempts = 1000;
  molecule_group = 0;
  molecule_group_bit = 0;
  molecule_group_inversebit = 0;
  exclusion_group = 0;
  exclusion_group_bit = 0;
  pressure_flag = false;
  pressure = 0.0;
  fugacity_coeff = 1.0;
  shakeflag = 0;
  charge = 0.0;
  charge_flag = false;
  full_flag = false;
  idshake = NULL;
  ngroups = 0;
  int ngroupsmax = 0;
  groupstrings = NULL;
  ngrouptypes = 0;
  int ngrouptypesmax = 0;
  grouptypestrings = NULL;
  grouptypes = NULL;
  grouptypebits = NULL;
  energy_intra = 0.0;
  tfac_insert = 1.0;

  int iarg = 0;
  while (iarg < narg) {
  if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1)
        error->all(FLERR,"Molecule template ID for fix gcmc does not exist");
      if (atom->molecules[imol]->nset > 1 && comm->me == 0)
        error->warning(FLERR,"Molecule template for "
                       "fix gcmc has multiple molecules");
      mode = MOLECULE;
      onemols = atom->molecules;
      nmol = onemols[imol]->nset;
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix gcmc does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxangle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      max_rotation_angle = force->numeric(FLERR,arg[iarg+1]);
      max_rotation_angle *= MY_PI/180;
      iarg += 2;
    } else if (strcmp(arg[iarg],"pair") == 0) { // added by Jibao. from Matias
        if (iarg+2 > narg) error->all(FLERR,"Illegal fix GCMC command");
        if (strcmp(arg[iarg+1],"lj/cut") == 0) pairflag = 0;
        else if (strcmp(arg[iarg+1],"Stw") == 0) pairflag = 1;
        else error->all(FLERR,"Illegal fix evaporate command");
        iarg += 2;
    }   // Matias
    else if (strcmp(arg[iarg],"pressure") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      pressure = force->numeric(FLERR,arg[iarg+1]);
        pressure = pressure * 100.0;    // added by Jibao, according to Matias' code
      pressure_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"fugacity_coeff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      fugacity_coeff = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"charge") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      charge = force->numeric(FLERR,arg[iarg+1]);
      charge_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      int n = strlen(arg[iarg+1]) + 1;
      delete [] idshake;
      idshake = new char[n];
      strcpy(idshake,arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"full_energy") == 0) {
      full_flag = true;
      iarg += 1;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngroups >= ngroupsmax) {
	ngroupsmax = ngroups+1;
	groupstrings = (char **)
	  memory->srealloc(groupstrings,
			   ngroupsmax*sizeof(char *),
			   "fix_gcmc:groupstrings");
      }
      int n = strlen(arg[iarg+1]) + 1;
      groupstrings[ngroups] = new char[n];
      strcpy(groupstrings[ngroups],arg[iarg+1]);
      ngroups++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"grouptype") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngrouptypes >= ngrouptypesmax) {
	ngrouptypesmax = ngrouptypes+1;
	grouptypes = (int*) memory->srealloc(grouptypes,ngrouptypesmax*sizeof(int),
			 "fix_gcmc:grouptypes");
	grouptypestrings = (char**)
	  memory->srealloc(grouptypestrings,
			   ngrouptypesmax*sizeof(char *),
			   "fix_gcmc:grouptypestrings");
      }
      grouptypes[ngrouptypes] = atoi(arg[iarg+1]);
      int n = strlen(arg[iarg+2]) + 1;
      grouptypestrings[ngrouptypes] = new char[n];
      strcpy(grouptypestrings[ngrouptypes],arg[iarg+2]);
      ngrouptypes++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"intra_energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      energy_intra = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tfac_insert") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      tfac_insert = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix gcmc command");
  }
}

/* ---------------------------------------------------------------------- */

FixGCMCVp::~FixGCMCVp()
{
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;
    
    //delete region_insert;   // from Matias; deleted by Jibao

  memory->destroy(local_gas_list);
  memory->destroy(atom_coord);
  memory->destroy(coords);
  memory->destroy(imageflags);

  delete [] idshake;

  if (ngroups > 0) {
    for (int igroup = 0; igroup < ngroups; igroup++)
      delete [] groupstrings[igroup];
    memory->sfree(groupstrings);
  }

  if (ngrouptypes > 0) {
    memory->destroy(grouptypes);
    memory->destroy(grouptypebits);
    for (int igroup = 0; igroup < ngrouptypes; igroup++)
      delete [] grouptypestrings[igroup];
    memory->sfree(grouptypestrings);
  }
}

/* ---------------------------------------------------------------------- */

int FixGCMCVp::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGCMCVp::init()
{
    
    //if (comm->me == 0) printf("Begins: FixGCMCVp::init()\n");

  triclinic = domain->triclinic;

  // decide whether to switch to the full_energy option

  if (!full_flag) {
    if ((force->kspace) ||
        (force->pair == NULL) ||
        (force->pair->single_enable == 0) ||
        (force->pair_match("hybrid",0)) ||
        (force->pair_match("eam",0))
	) {
      full_flag = true;
        
        //if (comm->me == 0) printf("Begins: inside if (!full_flag){}: FixGCMCVp::init()\n");
        
        //if (comm->me == 0) printf("pairflag = %d\n",pairflag);
        
        if (pairflag) { // added by Jibao
            full_flag = false;  // added by Jibao
        }   // added by Jibao
        
      if (comm->me == 0 && full_flag == true) // modified by Jibao
          error->warning(FLERR,"Fix gcmc using full_energy option");
    }
  }
    
    //if (comm->me == 0) printf("Begins 2: FixGCMCVp::init()\n");
    
  if (full_flag) {
    char *id_pe = (char *) "thermo_pe";
    int ipe = modify->find_compute(id_pe);
    c_pe = modify->compute[ipe];
  }

  int *type = atom->type;

  if (mode == ATOM) {
    if (ngcmc_type <= 0 || ngcmc_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
  }
    
    //if (comm->me == 0) printf("Begins 3: FixGCMCVp::init()\n");

  // if mode == ATOM, warn if any deletable atom has a mol ID

  if ((mode == ATOM) && atom->molecule_flag) {
      /*
      if (comm->me == 0) {
          printf("Inside if ((mode == ATOM)): FixGCMCVp::init()\n");
          printf("atom->molecule_flag = %d\n",atom->molecule_flag);
      }
      */
    tagint *molecule = atom->molecule;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == ngcmc_type)
        if (molecule[i]) flag = 1;
      
      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 2: FixGCMCVp::init()\n");
      
    int flagall;
      
      //printf("comm->me = %d, flag = %d, flagall = %d, before MPI_ALLreduce()\n",comm->me,flag,flagall);
      
      //error->all(FLERR,"Kao 0 !!!!");    // added by Jibao
      
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
      
      //error->all(FLERR,"Kao 1 !!!!");    // added by Jibao
      
      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 3: FixGCMCVp::init()\n");
      //if (comm->me == 0) printf("flag = %d, flagall = %d, after MPI_ALLreduce()\n",flag,flagall);
      
      if (flagall && comm->me == 0) {
          //if (comm->me == 0) printf("Inside if if (flagall && comm->me == 0): FixGCMCVp::init()\n");    // added by Jibao
          //error->all(FLERR,"Kao 2 !!!!");    // added by Jibao
          error->all(FLERR,"Fix gcmc cannot exchange individual atoms belonging to a molecule");
      }
  }
    
    //if (comm->me == 0) printf("Begins 4: FixGCMCVp::init()\n");

  // if mode == MOLECULE, check for unset mol IDs

  if (mode == MOLECULE) {
    tagint *molecule = atom->molecule;
    int *mask = atom->mask;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (mask[i] == groupbit)
        if (molecule[i] == 0) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall && comm->me == 0)
      error->all(FLERR,
       "All mol IDs should be set for fix gcmc group atoms");
  }
    
    //if (comm->me == 0) printf("Begins 5: FixGCMCVp::init()\n");

  if (((mode == MOLECULE) && (atom->molecule_flag == 0)) ||
      ((mode == MOLECULE) && (!atom->tag_enable || !atom->map_style)))
    error->all(FLERR,
               "Fix gcmc molecule command requires that "
               "atoms have molecule attributes");

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one
    
    //if (comm->me == 0) printf("Begins 6: FixGCMCVp::init()\n");

  fixshake = NULL;
  if (shakeflag) {
    int ifix = modify->find_fix(idshake);
    if (ifix < 0) error->all(FLERR,"Fix gcmc shake fix does not exist");
    fixshake = modify->fix[ifix];
    int tmp;
    if (onemols != (Molecule **) fixshake->extract("onemol",tmp))
      error->all(FLERR,"Fix gcmc and fix shake not using "
                 "same molecule template ID");
  }
    
    //if (comm->me == 0) printf("Begins 7: FixGCMCVp::init()\n");

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix gcmc in a 2d simulation");

  // create a new group for interaction exclusions
    
    //if (comm->me == 0) printf("Before 'create a new group for interaction exclusions': FixGCMCVp::init()\n");
    
  if (full_flag || pairflag) {  // modified by Jibao; added "|| pairflag"
    char **group_arg = new char*[4];
    // create unique group name for atoms to be excluded
    int len = strlen(id) + 30;
    group_arg[0] = new char[len];
    sprintf(group_arg[0],"FixGCMC:gcmc_exclusion_group:%s",id);
    group_arg[1] = (char *) "subtract";
    group_arg[2] = (char *) "all";
    group_arg[3] = (char *) "all";
    group->assign(4,group_arg);
    exclusion_group = group->find(group_arg[0]);
    if (exclusion_group == -1)
      error->all(FLERR,"Could not find fix gcmc exclusion group ID");
    exclusion_group_bit = group->bitmask[exclusion_group];

    // neighbor list exclusion setup
    // turn off interactions between group all and the exclusion group

    int narg = 4;
    char **arg = new char*[narg];;
    arg[0] = (char *) "exclude";
    arg[1] = (char *) "group";
    arg[2] = group_arg[0];
    arg[3] = (char *) "all";
    neighbor->modify_params(narg,arg);
    delete [] group_arg[0];
    delete [] group_arg;
    delete [] arg;
  }

  // create a new group for temporary use with selected molecules

  if (mode == MOLECULE) {
    char **group_arg = new char*[3];
    // create unique group name for atoms to be rotated
    int len = strlen(id) + 30;
    group_arg[0] = new char[len];
    sprintf(group_arg[0],"FixGCMC:rotation_gas_atoms:%s",id);
    group_arg[1] = (char *) "molecule";
    char digits[12];
    sprintf(digits,"%d",-1);
    group_arg[2] = digits;
    group->assign(3,group_arg);
    molecule_group = group->find(group_arg[0]);
    if (molecule_group == -1)
      error->all(FLERR,"Could not find fix gcmc rotation group ID");
    molecule_group_bit = group->bitmask[molecule_group];
    molecule_group_inversebit = molecule_group_bit ^ ~0;
    delete [] group_arg[0];
    delete [] group_arg;
  }

  // get all of the needed molecule data if mode == MOLECULE,
  // otherwise just get the gas mass

  if (mode == MOLECULE) {

    onemols[imol]->compute_mass();
    onemols[imol]->compute_com();
    gas_mass = onemols[imol]->masstotal;
    for (int i = 0; i < onemols[imol]->natoms; i++) {
      onemols[imol]->x[i][0] -= onemols[imol]->com[0];
      onemols[imol]->x[i][1] -= onemols[imol]->com[1];
      onemols[imol]->x[i][2] -= onemols[imol]->com[2];
    }

  } else gas_mass = atom->mass[ngcmc_type];

  if (gas_mass <= 0.0)
    error->all(FLERR,"Illegal fix gcmc gas mass <= 0");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do GCMC on atoms in atom_modify first group");
  }

    //if (comm->me == 0) printf("Before 'compute beta, lambda, sigma, and the zz factor': FixGCMCVp::init()\n");
    
  // compute beta, lambda, sigma, and the zz factor

  beta = 1.0/(force->boltz*reservoir_temperature);
  double lambda = sqrt(force->hplanck*force->hplanck/
                       (2.0*MY_PI*gas_mass*force->mvv2e*
                        force->boltz*reservoir_temperature));
  sigma = sqrt(force->boltz*reservoir_temperature*tfac_insert/gas_mass/force->mvv2e);
    
    if (!pressure_flag) // using pressure_flag to replace pressflag defined by Matias
        zz = exp(beta*chemical_potential)/(pow(lambda,3.0));
    else if(pressure_flag) {
        zz = pressure/(force->boltz*4.184*reservoir_temperature*1000/6.02e23)/(1e30);   // from Matias; need to check the meaning
        //zz = pressure*fugacity_coeff*beta/force->nktv2p;    // from the original expression of 2015 version of lammps
    }
    
    
  //zz = exp(beta*chemical_potential)/(pow(lambda,3.0)); // commented out by Jibao
  //if (pressure_flag) zz = pressure*fugacity_coeff*beta/force->nktv2p; // commented out by Jibao
    
    if (comm->me==0) { // added by Jibao; from Matias' version of lammps
        printf("zz factor equals %e\n",zz); // added by Jibao; from Matias' version of lammps
        printf("regionflag equals %i\n",regionflag); // added by Jibao; from Matias' version of lammps
        printf("pressure_flag equals %i\n",pressure_flag); // added by Jibao; from Matias' version of lammps
        printf("pairflag equals %i\n",pairflag); // added by Jibao; from Matias' version of lammps
    } // added by Jibao; from Matias' version of lammps

  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // construct group bitmask for all new atoms
  // aggregated over all group keywords

  groupbitall = 1 | groupbit;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    int jgroup = group->find(groupstrings[igroup]);
    if (jgroup == -1)
      error->all(FLERR,"Could not find specified fix gcmc group ID");
    groupbitall |= group->bitmask[jgroup];
  }

  // construct group type bitmasks
  // not aggregated over all group keywords

  if (ngrouptypes > 0) {
    memory->create(grouptypebits,ngrouptypes,"fix_gcmc:grouptypebits");
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      int jgroup = group->find(grouptypestrings[igroup]);
      if (jgroup == -1)
	error->all(FLERR,"Could not find specified fix gcmc group ID");
      grouptypebits[igroup] = group->bitmask[jgroup];
    }
  }
    
    //if (comm->me == 0) printf("End of FixGCMCVp::init()\n");

}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixGCMCVp::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  if (triclinic) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }

  if (regionflag) volume = region_volume;
  else volume = domain->xprd * domain->yprd * domain->zprd;

  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  update_gas_atoms_list();

  if (full_flag) {
    energy_stored = energy_full();

    if (mode == MOLECULE) {
      for (int i = 0; i < ncycles; i++) {
        int random_int_fraction =
          static_cast<int>(random_equal->uniform()*ncycles) + 1;
        if (random_int_fraction <= nmcmoves) {
 	  if (random_equal->uniform() < 0.5) attempt_molecule_translation_full();
 	  else attempt_molecule_rotation_full();
        } else {
          if (random_equal->uniform() < 0.5) attempt_molecule_deletion_full();
          else attempt_molecule_insertion_full();
        }
      }
    } else {
      for (int i = 0; i < ncycles; i++) {
        int random_int_fraction =
          static_cast<int>(random_equal->uniform()*ncycles) + 1;
        if (random_int_fraction <= nmcmoves) {
          attempt_atomic_translation_full();
        } else {
          if (random_equal->uniform() < 0.5) attempt_atomic_deletion_full();
          else attempt_atomic_insertion_full();
        }
      }
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

  } else {

    if (mode == MOLECULE) {
      for (int i = 0; i < ncycles; i++) {
        int random_int_fraction =
          static_cast<int>(random_equal->uniform()*ncycles) + 1;
        if (random_int_fraction <= nmcmoves) {
          if (random_equal->uniform() < 0.5) attempt_molecule_translation();
          else attempt_molecule_rotation();
        } else {
          if (random_equal->uniform() < 0.5) attempt_molecule_deletion();
          else attempt_molecule_insertion();
        }
      }
    } else {
      for (int i = 0; i < ncycles; i++) {
        int random_int_fraction =
          static_cast<int>(random_equal->uniform()*ncycles) + 1;
        if (random_int_fraction <= nmcmoves) {
          attempt_atomic_translation();
        } else {
            double tmp_debug = -2;
            tmp_debug = random_equal->uniform();
          if (tmp_debug < 0.5) attempt_atomic_deletion();   // modified by Jibao
          else attempt_atomic_insertion();
            
            //printf("comm->me = %d: random_equal->uniform()= %f, inside FixGCMCVp::pre_exchange()\n",comm->me,tmp_debug);
        }
      }
    }
      //domain->pbc();    // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
      //comm->exchange(); // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
  }
  next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_translation()
{
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  int i = pick_random_gas_atom();

  int success = 0;
  if (i >= 0) {
    double **x = atom->x;
    double energy_before = energy(i,ngcmc_type,-1,x[i]);
    double rsq = 1.1;
    double rx,ry,rz;
    rx = ry = rz = 0.0;
    double coord[3];
    while (rsq > 1.0) {
      rx = 2*random_unequal->uniform() - 1.0;
      ry = 2*random_unequal->uniform() - 1.0;
      rz = 2*random_unequal->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    }
    coord[0] = x[i][0] + displace*rx;
    coord[1] = x[i][1] + displace*ry;
    coord[2] = x[i][2] + displace*rz;
    if (regionflag) {
      while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
        rsq = 1.1;
        while (rsq > 1.0) {
          rx = 2*random_unequal->uniform() - 1.0;
          ry = 2*random_unequal->uniform() - 1.0;
          rz = 2*random_unequal->uniform() - 1.0;
          rsq = rx*rx + ry*ry + rz*rz;
        }
        coord[0] = x[i][0] + displace*rx;
        coord[1] = x[i][1] + displace*ry;
        coord[2] = x[i][2] + displace*rz;
      }
    }
    if (!domain->inside_nonperiodic(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");

    double energy_after = energy(i,ngcmc_type,-1,coord);
    if (random_unequal->uniform() <
        exp(beta*(energy_before - energy_after))) {
      x[i][0] = coord[0];
      x[i][1] = coord[1];
      x[i][2] = coord[2];
      success = 1;
    }
  }

  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  if (success_all) {
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    ntranslation_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_deletion()
{
    
    //if (comm->me == 0) printf("Beginning of FixGCMCVp::attempt_atomic_deletion()\n");
    
  ndeletion_attempts += 1.0;

  if (ngas == 0) return;

  int i = pick_random_gas_atom();

  int success = 0;
    
    double energy_all=0;  // Matias
    int proc_id=-2;       // Matias
    double deletion_energy; // added by Jibao
    
    //printf("comm->me = %d: i = %d, before if (i >= 0) in FixGCMCVp::attempt_atomic_deletion()\n",comm->me,i);
    
  if (i >= 0) {
      if (atom->type[i] != ngcmc_type) {
          printf("you are trying to delete an atom of type different from the one specified in fix gcmc command\natom->type[i=%d] = %d ngcmc_type = %d\n",i,atom->type[i],ngcmc_type); // added by Jibao
          error->all(FLERR,"Exit due to the error explained above");
      }
      
      
      if (pairflag == 0) {
          deletion_energy = energy(i,ngcmc_type,-1,atom->x[i]);
      } else if (pairflag == 1) {
          pair=force->pair; //force obtejo que tiene pair           // copied by Jibao from Matias
          //deletion_energy=pairsw->Stw_GCMC(i,ngcmc_type,1,atom->x[i]);   // copied by Jibao from Matias
          //printf("comm->me = %d, deletion_energy=pairsw->Stw_GCMC() = %f, i = %d\n",comm->me,deletion_energy,i);
          //deletion_energy = pair->Stw_GCMC(i,ngcmc_type,1,atom->x[i]);  // added by Jibao
          //printf("comm->me = %d, deletion_energy=pair->Stw_GCMC() = %f, i = %d\n",comm->me,deletion_energy,i);
          
          //printf("comm->me = %d: Before energy() in FixGCMCVp::attempt_atomic_deletion()\n",comm->me);
          deletion_energy = energy(i,ngcmc_type,-1,atom->x[i]);
          //printf("comm->me = %d, deletion_energy=energy() = %f, i = %d\n",comm->me,deletion_energy,i);
      }
      
      energy_all = deletion_energy; // Matias
      proc_id = comm->me; // Matias
      
      
    //double deletion_energy = energy(i,ngcmc_type,-1,atom->x[i]);  // commented out by Jibao
    if (random_unequal->uniform() <
        ngas*exp(beta*deletion_energy)/(zz*volume)) {
      atom->avec->copy(atom->nlocal-1,i,1);
      atom->nlocal--;
      success = 1;
    }
  }
    
    //printf("comm->me = %d: success = %d in FixGCMCVp::attempt_atomic_deletion()\n",comm->me,success);

  int success_all = 0;
    
    int proc_end = 0;   // from Matias; added by Jibao
    MPI_Barrier(world); // from Matias; added by Jibao
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);
    MPI_Allreduce(&proc_id,&proc_end,1,MPI_INT,MPI_MAX,world);  // from Matias; added by Jibao
    MPI_Bcast(&energy_all,1,MPI_DOUBLE,proc_end,world); // from Matias; added by Jibao
    energyout = energy_all; // from Matias; added by Jibao

  if (success_all) {
      // there was a "ngas--;" in version 2012; need to figure out where this is done in this version; commentted by Jibao
      //printf("comm->me = %d: deletion succeed\n",comm->me); // added by Jibao
    atom->natoms--;
    if (atom->tag_enable) {
      if (atom->map_style) atom->map_init();
    }
    atom->nghost = 0;
    comm->borders();
    update_gas_atoms_list();
    ndeletion_successes += 1.0; // == ndel_successes in version 2012; commentted by Jibao
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_insertion()
{
  double lamda[3];

  ninsertion_attempts += 1.0;

  // pick coordinates for insertion point

  double coord[3];
  if (regionflag) {
    int region_attempt = 0;
    coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
    coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
    coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
    while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(coord,lamda);
  } else {
    if (triclinic == 0) {
      coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,coord);
    }
  }

  int proc_flag = 0;
  if (triclinic == 0) {
    domain->remap(coord);
    if (!domain->inside(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");
    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
	coord[1] >= sublo[1] && coord[1] < subhi[1] &&
	coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
  } else {
    if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
	lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
	lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
  }

  int success = 0;
    
    double energy_all=0;  // from Matias; added by Jibao
    int proc_id=-2;       // from Matias; added by Jibao
    double insertion_energy; // added by Jibao
    
    //printf("comm->me = %d: proc_flag = %d, before if (proc_flag) in FixGCMCVp::attempt_atomic_insertion()\n",comm->me,proc_flag);   // added by Jibao
    
  if (proc_flag) {
      
      int nall = atom->nlocal + atom->nghost;   // from original 10Feb12 lammps version; added by Jibao
      
    int ii = -1;
    if (charge_flag) {
      ii = atom->nlocal + atom->nghost;
      if (ii >= atom->nmax) atom->avec->grow(0);
      atom->q[ii] = charge;
    }
    //double insertion_energy = energy(ii,ngcmc_type,-1,coord); // commented out by Jibao
      
      if (!pairflag) {  // from Matias; added by Jibao
          insertion_energy = energy(ii,ngcmc_type,-1,coord);    // from version 2015; added by Jibao
      } else if (pairflag) {    // from Matias; added by Jibao
          pair=force->pair; // from Matias; added by Jibao
          //insertion_energy = pairsw -> Stw_GCMC(nall,ngcmc_type,1,coord);    // from Matias; added by Jibao
          //insertion_energy = pair->Stw_GCMC(nall,ngcmc_type,1,coord);  // added by Jibao
          
          insertion_energy = energy(ii,ngcmc_type,-1,coord);
          //printf("comm->me = %d, insertion_energy=energy() = %f\n",comm->me,insertion_energy);
      }
      energy_all=insertion_energy;  // from Matias; added by Jibao
      proc_id=comm->me; // from Matias; added by Jibao
      
    if (random_unequal->uniform() <
        zz*volume*exp(-beta*insertion_energy)/(ngas+1)) {
      atom->avec->create_atom(ngcmc_type,coord);
      int m = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[m] = groupbitall;
      for (int igroup = 0; igroup < ngrouptypes; igroup++) {
	if (ngcmc_type == grouptypes[igroup])
	  atom->mask[m] |= grouptypebits[igroup];
      }

      atom->v[m][0] = random_unequal->gaussian()*sigma;
      atom->v[m][1] = random_unequal->gaussian()*sigma;
      atom->v[m][2] = random_unequal->gaussian()*sigma;
      modify->create_attribute(m);

      success = 1;
    }
  }
    
    //printf("comm->me = %d: success = %d in FixGCMCVp::attempt_atomic_insertion()\n",comm->me,success);   // added by Jibao

  int success_all = 0;
    
    
    int proc_end =0;    // from Matias; added by Jibao
    MPI_Barrier(world); // from Matias; added by Jibao
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);
    MPI_Allreduce(&proc_id,&proc_end,1,MPI_INT,MPI_MAX,world);  // from Matias; added by Jibao
    MPI_Bcast(&energy_all,1,MPI_DOUBLE,proc_end,world); // from Matias; added by Jibao
    energyout=energy_all;   // from Matias; added by Jibao
    
    //printf("comm->me = %d: Before insertion succeed\n",comm->me);    // added by Jibao
  if (success_all) {
      //printf("comm->me = %d: insertion succeed\n",comm->me);    // added by Jibao
    atom->natoms++;
    if (atom->tag_enable) {
      atom->tag_extend();
      if (atom->map_style) atom->map_init();
    }
    atom->nghost = 0;
    comm->borders();
    update_gas_atoms_list();
    ninsertion_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_translation()
{
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  tagint translation_molecule = pick_random_gas_molecule();
  if (translation_molecule == -1) return;

  double energy_before_sum = molecule_energy(translation_molecule);

  double **x = atom->x;
  double rx,ry,rz;
  double com_displace[3],coord[3];
  double rsq = 1.1;
  while (rsq > 1.0) {
    rx = 2*random_equal->uniform() - 1.0;
    ry = 2*random_equal->uniform() - 1.0;
    rz = 2*random_equal->uniform() - 1.0;
    rsq = rx*rx + ry*ry + rz*rz;
  }
  com_displace[0] = displace*rx;
  com_displace[1] = displace*ry;
  com_displace[2] = displace*rz;

  int nlocal = atom->nlocal;
  if (regionflag) {
    int *mask = atom->mask;
    for (int i = 0; i < nlocal; i++) {
      if (atom->molecule[i] == translation_molecule) {
        mask[i] |= molecule_group_bit;
      } else {
        mask[i] &= molecule_group_inversebit;
      }
    }
    double com[3];
    com[0] = com[1] = com[2] = 0.0;
    group->xcm(molecule_group,gas_mass,com);
    coord[0] = com[0] + displace*rx;
    coord[1] = com[1] + displace*ry;
    coord[2] = com[2] + displace*rz;
    while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
      rsq = 1.1;
      while (rsq > 1.0) {
        rx = 2*random_equal->uniform() - 1.0;
        ry = 2*random_equal->uniform() - 1.0;
        rz = 2*random_equal->uniform() - 1.0;
        rsq = rx*rx + ry*ry + rz*rz;
      }
      coord[0] = com[0] + displace*rx;
      coord[1] = com[1] + displace*ry;
      coord[2] = com[2] + displace*rz;
    }
    com_displace[0] = displace*rx;
    com_displace[1] = displace*ry;
    com_displace[2] = displace*rz;
  }

  double energy_after = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] == translation_molecule) {
      coord[0] = x[i][0] + com_displace[0];
      coord[1] = x[i][1] + com_displace[1];
      coord[2] = x[i][2] + com_displace[2];
      if (!domain->inside_nonperiodic(coord))
  	error->one(FLERR,"Fix gcmc put atom outside box");
      energy_after += energy(i,atom->type[i],translation_molecule,coord);
    }
  }

  double energy_after_sum = 0.0;
  MPI_Allreduce(&energy_after,&energy_after_sum,1,MPI_DOUBLE,MPI_SUM,world);

  if (random_equal->uniform() <
      exp(beta*(energy_before_sum - energy_after_sum))) {
    for (int i = 0; i < nlocal; i++) {
      if (atom->molecule[i] == translation_molecule) {
        x[i][0] += com_displace[0];
        x[i][1] += com_displace[1];
        x[i][2] += com_displace[2];
      }
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    ntranslation_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_rotation()
{
  nrotation_attempts += 1.0;

  if (ngas == 0) return;

  tagint rotation_molecule = pick_random_gas_molecule();
  if (rotation_molecule == -1) return;

  double energy_before_sum = molecule_energy(rotation_molecule);

  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] == rotation_molecule) {
      mask[i] |= molecule_group_bit;
    } else {
      mask[i] &= molecule_group_inversebit;
    }
  }

  double com[3];
  com[0] = com[1] = com[2] = 0.0;
  group->xcm(molecule_group,gas_mass,com);

  // generate point in unit cube
  // then restrict to unit sphere

  double r[3],rotmat[3][3],quat[4];
  double rsq = 1.1;
  while (rsq > 1.0) {
    r[0] = 2.0*random_equal->uniform() - 1.0;
    r[1] = 2.0*random_equal->uniform() - 1.0;
    r[2] = 2.0*random_equal->uniform() - 1.0;
    rsq = MathExtra::dot3(r, r);
  }

  double theta = random_equal->uniform() * max_rotation_angle;
  MathExtra::norm3(r);
  MathExtra::axisangle_to_quat(r,theta,quat);
  MathExtra::quat_to_mat(quat,rotmat);

  double **x = atom->x;
  imageint *image = atom->image;
  double energy_after = 0.0;
  int n = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & molecule_group_bit) {
      double xtmp[3];
      domain->unmap(x[i],image[i],xtmp);
      xtmp[0] -= com[0];
      xtmp[1] -= com[1];
      xtmp[2] -= com[2];
      MathExtra::matvec(rotmat,xtmp,atom_coord[n]);
      atom_coord[n][0] += com[0];
      atom_coord[n][1] += com[1];
      atom_coord[n][2] += com[2];
      xtmp[0] = atom_coord[n][0];
      xtmp[1] = atom_coord[n][1];
      xtmp[2] = atom_coord[n][2];
      domain->remap(xtmp);
      if (!domain->inside(xtmp))
	error->one(FLERR,"Fix gcmc put atom outside box");
      energy_after += energy(i,atom->type[i],rotation_molecule,xtmp);
      n++;
    }
  }

  double energy_after_sum = 0.0;
  MPI_Allreduce(&energy_after,&energy_after_sum,1,MPI_DOUBLE,MPI_SUM,world);

  if (random_equal->uniform() <
      exp(beta*(energy_before_sum - energy_after_sum))) {
    int n = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & molecule_group_bit) {
        image[i] = imagezero;
        x[i][0] = atom_coord[n][0];
        x[i][1] = atom_coord[n][1];
        x[i][2] = atom_coord[n][2];
        domain->remap(x[i],image[i]);
        n++;
      }
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    nrotation_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_deletion()
{
  ndeletion_attempts += 1.0;

  if (ngas == 0) return;

  tagint deletion_molecule = pick_random_gas_molecule();
  if (deletion_molecule == -1) return;

  double deletion_energy_sum = molecule_energy(deletion_molecule);

  if (random_equal->uniform() <
      ngas*exp(beta*deletion_energy_sum)/(zz*volume*natoms_per_molecule)) {
    int i = 0;
    while (i < atom->nlocal) {
      if (atom->molecule[i] == deletion_molecule) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      } else i++;
    }
    atom->natoms -= natoms_per_molecule;
    if (atom->map_style) atom->map_init();
    atom->nghost = 0;
    comm->borders();
    update_gas_atoms_list();
    ndeletion_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_insertion()
{
  double lamda[3];
  ninsertion_attempts += 1.0;

  double com_coord[3];
  if (regionflag) {
    int region_attempt = 0;
    com_coord[0] = region_xlo + random_equal->uniform() *
      (region_xhi-region_xlo);
    com_coord[1] = region_ylo + random_equal->uniform() *
      (region_yhi-region_ylo);
    com_coord[2] = region_zlo + random_equal->uniform() *
      (region_zhi-region_zlo);
    while (domain->regions[iregion]->match(com_coord[0],com_coord[1],
                                           com_coord[2]) == 0) {
      com_coord[0] = region_xlo + random_equal->uniform() *
        (region_xhi-region_xlo);
      com_coord[1] = region_ylo + random_equal->uniform() *
        (region_yhi-region_ylo);
      com_coord[2] = region_zlo + random_equal->uniform() *
        (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(com_coord,lamda);
  } else {
    if (triclinic == 0) {
      com_coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      com_coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      com_coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,com_coord);
    }
  }

  // generate point in unit cube
  // then restrict to unit sphere

  double r[3],rotmat[3][3],quat[4];
  double rsq = 1.1;
  while (rsq > 1.0) {
    r[0] = 2.0*random_equal->uniform() - 1.0;
    r[1] = 2.0*random_equal->uniform() - 1.0;
    r[2] = 2.0*random_equal->uniform() - 1.0;
    rsq = MathExtra::dot3(r, r);
  }

  double theta = random_equal->uniform() * MY_2PI;
  MathExtra::norm3(r);
  MathExtra::axisangle_to_quat(r,theta,quat);
  MathExtra::quat_to_mat(quat,rotmat);

  double insertion_energy = 0.0;
  bool procflag[natoms_per_molecule];

  for (int i = 0; i < natoms_per_molecule; i++) {
    MathExtra::matvec(rotmat,onemols[imol]->x[i],atom_coord[i]);
    atom_coord[i][0] += com_coord[0];
    atom_coord[i][1] += com_coord[1];
    atom_coord[i][2] += com_coord[2];

    // use temporary variable for remapped position
    // so unmapped position is preserved in atom_coord

    double xtmp[3];
    xtmp[0] = atom_coord[i][0];
    xtmp[1] = atom_coord[i][1];
    xtmp[2] = atom_coord[i][2];
    domain->remap(xtmp);
    if (!domain->inside(xtmp))
      error->one(FLERR,"Fix gcmc put atom outside box");

    procflag[i] = false;
    if (triclinic == 0) {
      if (xtmp[0] >= sublo[0] && xtmp[0] < subhi[0] &&
	  xtmp[1] >= sublo[1] && xtmp[1] < subhi[1] &&
	  xtmp[2] >= sublo[2] && xtmp[2] < subhi[2]) procflag[i] = true;
    } else {
      domain->x2lamda(xtmp,lamda);
      if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
	  lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
	  lamda[2] >= sublo[2] && lamda[2] < subhi[2]) procflag[i] = true;
    }

    if (procflag[i]) {
      int ii = -1;
      if (onemols[imol]->qflag == 1) {
	ii = atom->nlocal + atom->nghost;
	if (ii >= atom->nmax) atom->avec->grow(0);
	atom->q[ii] = onemols[imol]->q[i];
      }
      insertion_energy += energy(ii,onemols[imol]->type[i],-1,xtmp);
    }
  }

  double insertion_energy_sum = 0.0;
  MPI_Allreduce(&insertion_energy,&insertion_energy_sum,1,
                MPI_DOUBLE,MPI_SUM,world);

  if (random_equal->uniform() < zz*volume*natoms_per_molecule*
      exp(-beta*insertion_energy_sum)/(ngas + natoms_per_molecule)) {

    tagint maxmol = 0;
    for (int i = 0; i < atom->nlocal; i++) maxmol = MAX(maxmol,atom->molecule[i]);
    tagint maxmol_all;
    MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
    maxmol_all++;
    if (maxmol_all >= MAXTAGINT)
      error->all(FLERR,"Fix gcmc ran out of available molecule IDs");

    tagint maxtag = 0;
    for (int i = 0; i < atom->nlocal; i++) maxtag = MAX(maxtag,atom->tag[i]);
    tagint maxtag_all;
    MPI_Allreduce(&maxtag,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

    int nlocalprev = atom->nlocal;

    double vnew[3];
    vnew[0] = random_equal->gaussian()*sigma;
    vnew[1] = random_equal->gaussian()*sigma;
    vnew[2] = random_equal->gaussian()*sigma;

    for (int i = 0; i < natoms_per_molecule; i++) {
      if (procflag[i]) {
        atom->avec->create_atom(onemols[imol]->type[i],atom_coord[i]);
        int m = atom->nlocal - 1;

	// add to groups
	// optionally add to type-based groups

	atom->mask[m] = groupbitall;
	for (int igroup = 0; igroup < ngrouptypes; igroup++) {
	  if (ngcmc_type == grouptypes[igroup])
	    atom->mask[m] |= grouptypebits[igroup];
	}

        atom->image[m] = imagezero;
        domain->remap(atom->x[m],atom->image[m]);
        atom->molecule[m] = maxmol_all;
        if (maxtag_all+i+1 >= MAXTAGINT)
          error->all(FLERR,"Fix gcmc ran out of available atom IDs");
        atom->tag[m] = maxtag_all + i + 1;
        atom->v[m][0] = vnew[0];
        atom->v[m][1] = vnew[1];
        atom->v[m][2] = vnew[2];

        atom->add_molecule_atom(onemols[imol],i,m,maxtag_all);
        modify->create_attribute(m);
      }
    }

    if (shakeflag)
      fixshake->set_molecule(nlocalprev,maxtag_all,imol,com_coord,vnew,quat);

    atom->natoms += natoms_per_molecule;
    if (atom->natoms < 0)
      error->all(FLERR,"Too many total atoms");
    atom->nbonds += onemols[imol]->nbonds;
    atom->nangles += onemols[imol]->nangles;
    atom->ndihedrals += onemols[imol]->ndihedrals;
    atom->nimpropers += onemols[imol]->nimpropers;
    if (atom->map_style) atom->map_init();
    atom->nghost = 0;
    comm->borders();
    update_gas_atoms_list();
    ninsertion_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_translation_full()
{
    error->all(FLERR,"Cannot be used for atomic translation"); // added by Jibao
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  double energy_before = energy_stored;

  int i = pick_random_gas_atom();

  double **x = atom->x;
  double xtmp[3];

  xtmp[0] = xtmp[1] = xtmp[2] = 0.0;

  tagint tmptag = -1;

  if (i >= 0) {

    double rsq = 1.1;
    double rx,ry,rz;
    rx = ry = rz = 0.0;
    double coord[3];
    while (rsq > 1.0) {
      rx = 2*random_unequal->uniform() - 1.0;
      ry = 2*random_unequal->uniform() - 1.0;
      rz = 2*random_unequal->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    }
    coord[0] = x[i][0] + displace*rx;
    coord[1] = x[i][1] + displace*ry;
    coord[2] = x[i][2] + displace*rz;
    if (regionflag) {
      while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
        rsq = 1.1;
        while (rsq > 1.0) {
          rx = 2*random_unequal->uniform() - 1.0;
          ry = 2*random_unequal->uniform() - 1.0;
          rz = 2*random_unequal->uniform() - 1.0;
          rsq = rx*rx + ry*ry + rz*rz;
        }
        coord[0] = x[i][0] + displace*rx;
        coord[1] = x[i][1] + displace*ry;
        coord[2] = x[i][2] + displace*rz;
      }
    }
    if (!domain->inside_nonperiodic(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");
    xtmp[0] = x[i][0];
    xtmp[1] = x[i][1];
    xtmp[2] = x[i][2];
    x[i][0] = coord[0];
    x[i][1] = coord[1];
    x[i][2] = coord[2];

    tmptag = atom->tag[i];
  }

  double energy_after = energy_full();

  if (random_equal->uniform() <
      exp(beta*(energy_before - energy_after))) {
    energy_stored = energy_after;
    ntranslation_successes += 1.0;
  } else {

    tagint tmptag_all;
    MPI_Allreduce(&tmptag,&tmptag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

    double xtmp_all[3];
    MPI_Allreduce(&xtmp,&xtmp_all,3,MPI_DOUBLE,MPI_SUM,world);

    for (int i = 0; i < atom->nlocal; i++) {
      if (tmptag_all == atom->tag[i]) {
        x[i][0] = xtmp_all[0];
        x[i][1] = xtmp_all[1];
        x[i][2] = xtmp_all[2];
      }
    }
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_deletion_full()
{
    double q_tmp;
    const int q_flag = atom->q_flag;
    
    ndeletion_attempts += 1.0;
    
    if (ngas == 0) return;
    
    double energy_before = energy_stored;
    
    const int i = pick_random_gas_atom();
    
    int tmpmask;
    if (i >= 0) {
        tmpmask = atom->mask[i];
        atom->mask[i] = exclusion_group_bit;
        if (q_flag) {
            q_tmp = atom->q[i];
            atom->q[i] = 0.0;
        }
    }
    if (force->kspace) force->kspace->qsum_qsq();
    double energy_after = energy_full();
    
    //printf("comm->me = %d, deletion_full: energy_before-energy_after = %f\n",comm->me,energy_before-energy_after); //added by Jibao
    
    double deletion_energy = energy(i,ngcmc_type,-1,atom->x[i]);
    //printf("comm->me = %d, deletion_energy=energy() = %f, i = %d\n",comm->me,deletion_energy,i);    // added by Jibao
    
    if (random_equal->uniform() <
        ngas*exp(beta*(energy_before - energy_after))/(zz*volume)) {
        if (i >= 0) {
            atom->avec->copy(atom->nlocal-1,i,1);
            atom->nlocal--;
        }
        atom->natoms--;
        if (atom->map_style) atom->map_init();
        ndeletion_successes += 1.0;
        energy_stored = energy_after;
    } else {
        if (i >= 0) {
            atom->mask[i] = tmpmask;
            if (q_flag) atom->q[i] = q_tmp;
        }
        if (force->kspace) force->kspace->qsum_qsq();
        energy_stored = energy_before;
    }
    update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_atomic_insertion_full()
{
  double lamda[3];
  ninsertion_attempts += 1.0;

  double energy_before = energy_stored;

  double coord[3];
  if (regionflag) {
    int region_attempt = 0;
    coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
    coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
    coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
    while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(coord,lamda);
  } else {
    if (triclinic == 0) {
      coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,coord);
    }
  }

  int proc_flag = 0;
  if (triclinic == 0) {
    domain->remap(coord);
    if (!domain->inside(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");
    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
	coord[1] >= sublo[1] && coord[1] < subhi[1] &&
	coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
  } else {
    if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
	lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
	lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
  }

  if (proc_flag) {
    atom->avec->create_atom(ngcmc_type,coord);
    int m = atom->nlocal - 1;

    // add to groups
    // optionally add to type-based groups

    atom->mask[m] = groupbitall;
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      if (ngcmc_type == grouptypes[igroup])
	atom->mask[m] |= grouptypebits[igroup];
    }

    atom->v[m][0] = random_unequal->gaussian()*sigma;
    atom->v[m][1] = random_unequal->gaussian()*sigma;
    atom->v[m][2] = random_unequal->gaussian()*sigma;
    if (charge_flag) atom->q[m] = charge;
    modify->create_attribute(m);
  }

  atom->natoms++;
  if (atom->tag_enable) {
    atom->tag_extend();
    if (atom->map_style) atom->map_init();
  }
  atom->nghost = 0;
  comm->borders();
  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  if (random_equal->uniform() <
      zz*volume*exp(beta*(energy_before - energy_after))/(ngas+1)) {

    ninsertion_successes += 1.0;
    energy_stored = energy_after;
  } else {
    atom->natoms--;
    if (proc_flag) atom->nlocal--;
    if (force->kspace) force->kspace->qsum_qsq();
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_translation_full()
{
    error->all(FLERR,"Cannot be used for molecule translation"); // added by Jibao
  ntranslation_attempts += 1.0;

  if (ngas == 0) return;

  tagint translation_molecule = pick_random_gas_molecule();
  if (translation_molecule == -1) return;

  double energy_before = energy_stored;

  double **x = atom->x;
  double rx,ry,rz;
  double com_displace[3],coord[3];
  double rsq = 1.1;
  while (rsq > 1.0) {
    rx = 2*random_equal->uniform() - 1.0;
    ry = 2*random_equal->uniform() - 1.0;
    rz = 2*random_equal->uniform() - 1.0;
    rsq = rx*rx + ry*ry + rz*rz;
  }
  com_displace[0] = displace*rx;
  com_displace[1] = displace*ry;
  com_displace[2] = displace*rz;

  int nlocal = atom->nlocal;
  if (regionflag) {
    int *mask = atom->mask;
    for (int i = 0; i < nlocal; i++) {
      if (atom->molecule[i] == translation_molecule) {
        mask[i] |= molecule_group_bit;
      } else {
        mask[i] &= molecule_group_inversebit;
      }
    }
    double com[3];
    com[0] = com[1] = com[2] = 0.0;
    group->xcm(molecule_group,gas_mass,com);
    coord[0] = com[0] + displace*rx;
    coord[1] = com[1] + displace*ry;
    coord[2] = com[2] + displace*rz;
    while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
      rsq = 1.1;
      while (rsq > 1.0) {
        rx = 2*random_equal->uniform() - 1.0;
        ry = 2*random_equal->uniform() - 1.0;
        rz = 2*random_equal->uniform() - 1.0;
        rsq = rx*rx + ry*ry + rz*rz;
      }
      coord[0] = com[0] + displace*rx;
      coord[1] = com[1] + displace*ry;
      coord[2] = com[2] + displace*rz;
    }
    com_displace[0] = displace*rx;
    com_displace[1] = displace*ry;
    com_displace[2] = displace*rz;
  }

  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] == translation_molecule) {
      x[i][0] += com_displace[0];
      x[i][1] += com_displace[1];
      x[i][2] += com_displace[2];
      if (!domain->inside_nonperiodic(x[i]))
	error->one(FLERR,"Fix gcmc put atom outside box");
    }
  }

  double energy_after = energy_full();

  if (random_equal->uniform() <
      exp(beta*(energy_before - energy_after))) {
    ntranslation_successes += 1.0;
    energy_stored = energy_after;
  } else {
    energy_stored = energy_before;
    for (int i = 0; i < nlocal; i++) {
      if (atom->molecule[i] == translation_molecule) {
        x[i][0] -= com_displace[0];
        x[i][1] -= com_displace[1];
        x[i][2] -= com_displace[2];
      }
    }
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_rotation_full()
{
    error->all(FLERR,"Cannot be used for molecule rotation"); // added by Jibao
  nrotation_attempts += 1.0;

  if (ngas == 0) return;

  tagint rotation_molecule = pick_random_gas_molecule();
  if (rotation_molecule == -1) return;

  double energy_before = energy_stored;

  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] == rotation_molecule) {
      mask[i] |= molecule_group_bit;
    } else {
      mask[i] &= molecule_group_inversebit;
    }
  }

  double com[3];
  com[0] = com[1] = com[2] = 0.0;
  group->xcm(molecule_group,gas_mass,com);

  // generate point in unit cube
  // then restrict to unit sphere

  double r[3],rotmat[3][3],quat[4];
  double rsq = 1.1;
  while (rsq > 1.0) {
    r[0] = 2.0*random_equal->uniform() - 1.0;
    r[1] = 2.0*random_equal->uniform() - 1.0;
    r[2] = 2.0*random_equal->uniform() - 1.0;
    rsq = MathExtra::dot3(r, r);
  }

  double theta = random_equal->uniform() * max_rotation_angle;
  MathExtra::norm3(r);
  MathExtra::axisangle_to_quat(r,theta,quat);
  MathExtra::quat_to_mat(quat,rotmat);

  double **x = atom->x;
  imageint *image = atom->image;
  imageint image_orig[natoms_per_molecule];
  int n = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & molecule_group_bit) {
      atom_coord[n][0] = x[i][0];
      atom_coord[n][1] = x[i][1];
      atom_coord[n][2] = x[i][2];
      image_orig[n] = image[i];
      double xtmp[3];
      domain->unmap(x[i],image[i],xtmp);
      xtmp[0] -= com[0];
      xtmp[1] -= com[1];
      xtmp[2] -= com[2];
      MathExtra::matvec(rotmat,xtmp,x[i]);
      x[i][0] += com[0];
      x[i][1] += com[1];
      x[i][2] += com[2];
      image[i] = imagezero;
      domain->remap(x[i],image[i]);
      if (!domain->inside(x[i]))
	error->one(FLERR,"Fix gcmc put atom outside box");
      n++;
    }
  }

  double energy_after = energy_full();

  if (random_equal->uniform() <
      exp(beta*(energy_before - energy_after))) {
    nrotation_successes += 1.0;
    energy_stored = energy_after;
  } else {
    energy_stored = energy_before;
    int n = 0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & molecule_group_bit) {
        x[i][0] = atom_coord[n][0];
        x[i][1] = atom_coord[n][1];
        x[i][2] = atom_coord[n][2];
        image[i] = image_orig[n];
        n++;
      }
    }
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_deletion_full()
{
    error->all(FLERR,"Cannot be used for molecule deletion"); // added by Jibao
  ndeletion_attempts += 1.0;

  if (ngas == 0) return;

  tagint deletion_molecule = pick_random_gas_molecule();
  if (deletion_molecule == -1) return;

  double energy_before = energy_stored;

  int m = 0;
  double q_tmp[natoms_per_molecule];
  int tmpmask[atom->nlocal];
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] == deletion_molecule) {
      tmpmask[i] = atom->mask[i];
      atom->mask[i] = exclusion_group_bit;
      toggle_intramolecular(i);
      if (atom->q_flag) {
        q_tmp[m] = atom->q[i];
        m++;
        atom->q[i] = 0.0;
      }
    }
  }
  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  // energy_before corrected by energy_intra

  double deltaphi = ngas*exp(beta*((energy_before - energy_intra) - energy_after))/(zz*volume*natoms_per_molecule);

  if (random_equal->uniform() < deltaphi) {
    int i = 0;
    while (i < atom->nlocal) {
      if (atom->molecule[i] == deletion_molecule) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      } else i++;
    }
    atom->natoms -= natoms_per_molecule;
    if (atom->map_style) atom->map_init();
    ndeletion_successes += 1.0;
    energy_stored = energy_after;
  } else {
    energy_stored = energy_before;
    int m = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->molecule[i] == deletion_molecule) {
        atom->mask[i] = tmpmask[i];
        toggle_intramolecular(i);
        if (atom->q_flag) {
          atom->q[i] = q_tmp[m];
          m++;
        }
      }
    }
    if (force->kspace) force->kspace->qsum_qsq();
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::attempt_molecule_insertion_full()
{
    error->all(FLERR,"Cannot be used for molecule insertion"); // added by Jibao
  double lamda[3];
  ninsertion_attempts += 1.0;

  double energy_before = energy_stored;

  tagint maxmol = 0;
  for (int i = 0; i < atom->nlocal; i++) maxmol = MAX(maxmol,atom->molecule[i]);
  tagint maxmol_all;
  MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
  maxmol_all++;
  if (maxmol_all >= MAXTAGINT)
    error->all(FLERR,"Fix gcmc ran out of available molecule IDs");
  int insertion_molecule = maxmol_all;

  tagint maxtag = 0;
  for (int i = 0; i < atom->nlocal; i++) maxtag = MAX(maxtag,atom->tag[i]);
  tagint maxtag_all;
  MPI_Allreduce(&maxtag,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

  int nlocalprev = atom->nlocal;

  double com_coord[3];
  if (regionflag) {
    int region_attempt = 0;
    com_coord[0] = region_xlo + random_equal->uniform() *
      (region_xhi-region_xlo);
    com_coord[1] = region_ylo + random_equal->uniform() *
      (region_yhi-region_ylo);
    com_coord[2] = region_zlo + random_equal->uniform() *
      (region_zhi-region_zlo);
    while (domain->regions[iregion]->match(com_coord[0],com_coord[1],
                                           com_coord[2]) == 0) {
      com_coord[0] = region_xlo + random_equal->uniform() *
        (region_xhi-region_xlo);
      com_coord[1] = region_ylo + random_equal->uniform() *
        (region_yhi-region_ylo);
      com_coord[2] = region_zlo + random_equal->uniform() *
        (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(com_coord,lamda);
  } else {
    if (triclinic == 0) {
      com_coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      com_coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      com_coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,com_coord);
    }

  }

  // generate point in unit cube
  // then restrict to unit sphere

  double r[3],rotmat[3][3],quat[4];
  double rsq = 1.1;
  while (rsq > 1.0) {
    r[0] = 2.0*random_equal->uniform() - 1.0;
    r[1] = 2.0*random_equal->uniform() - 1.0;
    r[2] = 2.0*random_equal->uniform() - 1.0;
    rsq = MathExtra::dot3(r, r);
  }

  double theta = random_equal->uniform() * MY_2PI;
  MathExtra::norm3(r);
  MathExtra::axisangle_to_quat(r,theta,quat);
  MathExtra::quat_to_mat(quat,rotmat);

  double vnew[3];
  vnew[0] = random_equal->gaussian()*sigma;
  vnew[1] = random_equal->gaussian()*sigma;
  vnew[2] = random_equal->gaussian()*sigma;

  for (int i = 0; i < natoms_per_molecule; i++) {
    double xtmp[3];
    MathExtra::matvec(rotmat,onemols[imol]->x[i],xtmp);
    xtmp[0] += com_coord[0];
    xtmp[1] += com_coord[1];
    xtmp[2] += com_coord[2];

    // need to adjust image flags in remap()

    imageint imagetmp = imagezero;
    domain->remap(xtmp,imagetmp);
    if (!domain->inside(xtmp))
      error->one(FLERR,"Fix gcmc put atom outside box");

    int proc_flag = 0;
    if (triclinic == 0) {
      if (xtmp[0] >= sublo[0] && xtmp[0] < subhi[0] &&
	  xtmp[1] >= sublo[1] && xtmp[1] < subhi[1] &&
	  xtmp[2] >= sublo[2] && xtmp[2] < subhi[2]) proc_flag = 1;
    } else {
      domain->x2lamda(xtmp,lamda);
      if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
	  lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
	  lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
    }

    if (proc_flag) {
      atom->avec->create_atom(onemols[imol]->type[i],xtmp);
      int m = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[m] = groupbitall;
      for (int igroup = 0; igroup < ngrouptypes; igroup++) {
	if (ngcmc_type == grouptypes[igroup])
	  atom->mask[m] |= grouptypebits[igroup];
      }

      atom->image[m] = imagetmp;
      atom->molecule[m] = insertion_molecule;
      if (maxtag_all+i+1 >= MAXTAGINT)
        error->all(FLERR,"Fix gcmc ran out of available atom IDs");
      atom->tag[m] = maxtag_all + i + 1;
      atom->v[m][0] = vnew[0];
      atom->v[m][1] = vnew[1];
      atom->v[m][2] = vnew[2];

      atom->add_molecule_atom(onemols[imol],i,m,maxtag_all);
      modify->create_attribute(m);
    }
  }

  if (shakeflag)
    fixshake->set_molecule(nlocalprev,maxtag_all,imol,com_coord,vnew,quat);

  atom->natoms += natoms_per_molecule;
  if (atom->natoms < 0)
    error->all(FLERR,"Too many total atoms");
  atom->nbonds += onemols[imol]->nbonds;
  atom->nangles += onemols[imol]->nangles;
  atom->ndihedrals += onemols[imol]->ndihedrals;
  atom->nimpropers += onemols[imol]->nimpropers;
  if (atom->map_style) atom->map_init();
  atom->nghost = 0;
  comm->borders();
  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  // energy_after corrected by energy_intra

  double deltaphi = zz*volume*natoms_per_molecule*
    exp(beta*(energy_before - (energy_after - energy_intra)))/(ngas + natoms_per_molecule);

  if (random_equal->uniform() < deltaphi) {

    ninsertion_successes += 1.0;
    energy_stored = energy_after;

  } else {

    atom->nbonds -= onemols[imol]->nbonds;
    atom->nangles -= onemols[imol]->nangles;
    atom->ndihedrals -= onemols[imol]->ndihedrals;
    atom->nimpropers -= onemols[imol]->nimpropers;
    atom->natoms -= natoms_per_molecule;

    energy_stored = energy_before;
    int i = 0;
    while (i < atom->nlocal) {
      if (atom->molecule[i] == insertion_molecule) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      } else i++;
    }
    if (force->kspace) force->kspace->qsum_qsq();
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   compute particle's interaction energy with the rest of the system
------------------------------------------------------------------------- */

double FixGCMCVp::energy(int i, int itype, tagint imolecule, double *coord)
{
    //printf("comm->me = %d: Beginning of FixGCMCVp::energy()\n",comm->me);
    double delx,dely,delz,rsq;
    double rsq1, rsq2;  // added by Jibao; for Stw_GCMC
    int ietype,jetype,ketype,ijparam,ikparam,ijkparam;  // added by Jibao; ietype: element type of itype
    int jtype,ktype;
    double delr1[3],delr2[3],fj[3],fk[3];   // added by Jibao; for Stw_GCMC
    int jj,kk;  // added by Jibao; for Stw_GCMC
    
    double **x = atom->x;
    int *type = atom->type;
    tagint *molecule = atom->molecule;
    int nall = atom->nlocal + atom->nghost;
    
    //printf("comm->me = %d: nall= %d, atom->nlocal= %d, atom->nghost= %d\n",comm->me,nall,atom->nlocal,atom->nghost);
    
    pair = force->pair;
    cutsq = force->pair->cutsq;
    
    double fpair = 0.0;
    double factor_coul = 1.0;
    double factor_lj = 1.0;
    
    double total_energy = 0.0;
    double twobodyeng = 0.0;    // added by Jibao; for Stw_GCMC
    double tmp2body = 0.0;      // added by Jibao; for Stw_GCMC
    double threebodyeng = 0.0;  // added by Jibao; for Stw_GCMC
    double tmp3body = 0.0;      // added by Jibao; for Stw_GCMC
    
    char *pair_style;   // added by Jibao
    pair_style = force->pair_style; // added by Jibao
    int eflag = 1;  // added by Jibao; for twobody() and threebody()
    
    //printf("comm->me = %d: 2 of FixGCMCVp::energy()\n",comm->me);
    
    if (!strcmp(pair_style,"hybrid")) {
        int ***map_substyle = (int ***) pair->returnmap_substyle();   // added by Jibao; return list of sub-styles itype,jtype points to
        
        Pair **styles = (Pair **) pair->returnstyles();     // added by Jibao; return list of Pair style classes
        
        char **keywords = (char **) pair->returnkeywords();   // added by Jibao; return style name of each Pair style
        
        //printf("comm->me = %d: 3 of FixGCMCVp::energy()\n",comm->me);
        
        for(int j=0; j < nall; j++){
            
            //printf("comm->me = %d: 3.1 of FixGCMCVp::energy()\n",comm->me);
            
            if (i == j) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[j]) continue;
            
            //printf("comm->me = %d: 3.2 of FixGCMCVp::energy()\n",comm->me);
            
            delx = coord[0] - x[j][0];
            dely = coord[1] - x[j][1];
            delz = coord[2] - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            
            //printf("comm->me = %d: 3.3 of FixGCMCVp::energy()\n",comm->me);
            
            //printf("comm->me = %d: j = %d\n",comm->me,j);
            
            jtype = type[j];
            
            //printf("comm->me = %d: type[j]= type[%d]= %d\n",comm->me,j,type[j]);
            
            //printf("comm->me = %d: 3.4 of FixGCMCVp::energy()\n",comm->me);
            
            int **nmap = (int **) pair->returnnmap();
            //printf("comm->me = %d: nmap[%d][%d]= %d\n",comm->me,itype,jtype,nmap[itype][jtype]);
            
            if (nmap[itype][jtype] > 0) {
                //printf("comm->me = %d: itype= %d,jtype= %d,map_substyle[%d][%d][0]= %d\n",comm->me,itype,jtype,itype,jtype,map_substyle[itype][jtype][0]);
                
                //printf("comm->me = %d: keywords[map_substyle[%d][%d][0]] = %s\n",comm->me,itype,jtype,keywords[map_substyle[itype][jtype][0]]);
                
                int substyle = map_substyle[itype][jtype][0];
                
                if (strstr(keywords[substyle],"sw")) {
                    
                    //if (!strcmp(keywords[substyle],"sw") || !strcmp(keywords[substyle],"sw0") || !strcmp(keywords[substyle],"sw/omp") || !strcmp(keywords[substyle],"sw0/omp")) {
                    
                    
                    //pair = force->pair_match("hybrid",0);
                    
                    //printf("comm->me = %d: 4 of FixGCMCVp::energy()\n",comm->me);
                    
                    int *map = (int *) styles[substyle]->returnmap();
                    //int *map = (int *) pair->returnmap();
                    
                    //printf("comm->me = %d: 4.1 of FixGCMCVp::energy()\n",comm->me);
                    //printf("itype= %d in FixGCMCVp::energy()\n",itype);
                    
                    //for (int kao = 1; kao<=7; kao++) printf("map[%d] = %d\n",kao,map[kao]);
                    
                    
                    //printf("map[itype] = map[%d] = %d in FixGCMCVp::energy()\n",itype,map[itype]);
                    
                    ietype=map[itype];
                    
                    //printf("comm->me = %d: 4.2 of FixGCMCVp::energy()\n",comm->me);
                    
                    LAMMPS_NS::Pair::Param *params = (LAMMPS_NS::Pair::Param *) styles[substyle]->returnparams();   // parameter set for an I-J-K interaction
                    //printf("comm->me = %d: 4.3 of FixGCMCVp::energy()\n",comm->me);
                    
                    int ***elem2param = (int ***) styles[substyle]->returnelem2param();
                    //printf("comm->me = %d: 4.4 of FixGCMCVp::energy()\n",comm->me);
                    
                    jetype=map[jtype];
                    //printf("comm->me = %d: 4.5 of FixGCMCVp::energy()\n",comm->me);
                    //printf("comm->me = %d: itype= %d, ietype= %d, jtype= %d, map[jtype]= %d\n",comm->me,itype,ietype, jtype, map[jtype]);
                    
                    ijparam = elem2param[ietype][jetype][jetype];
                    //printf("comm->me = %d: 4.6 of FixGCMCVp::energy()\n",comm->me);
                    
                    //printf("comm->me = %d: 5 of FixGCMCVp::energy()\n",comm->me);
                    
                    if (rsq < params[ijparam].cutsq){
                        styles[substyle]->twobody(&params[ijparam],rsq,fpair,eflag,tmp2body);
                        //printf("comm->me = %d: tmp2body = %f\n",comm->me,tmp2body);
                        total_energy+=tmp2body;
                        //printf("# %d %d %d %d %d %f %e\n",ietype,type[j],ietype,jetype,ijparam,params[ijparam].epsilon,tmp2body);
                    }
                    //printf("comm->me = %d: 6 of FixGCMCVp::energy()\n",comm->me);
                } else {
                    //printf("comm->me = %d: 7 of FixGCMCVp::energy()\n",comm->me);
                    if (rsq < cutsq[itype][jtype])
                        total_energy +=
                        styles[substyle]->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);
                }
            }
            //printf("comm->me = %d: 8 of FixGCMCVp::energy()\n",comm->me);
        }
        
        //printf("comm->me = %d: after twobody loop\n",comm->me);
        
        // only for sw, sw0 potential
        // first possibility ii!=i j==i k!=i or ii!=i; j!=i ; k==i
        for(int ii = 0;ii < nall;ii++){
            if (ii==i) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[ii]) continue;
            
            int **nmap = (int **) pair->returnnmap();
            //printf("first possibility: nmap[%d][%d]= %d in for(int ii = 0;ii < nall;ii++){}\n",jtype,itype,nmap[jtype][itype]);
            jtype = type[ii];
            
            if (nmap[jtype][itype] > 0) {
                
                int substyle_ji = map_substyle[jtype][itype][0];
                
                //printf("map_substyle[%d][%d][0]= %d in for(int ii = 0;ii < nall;ii++){}\n",jtype,itype,map_substyle[jtype][itype][0]);
                
                //if (comm->me == 0) printf("map_substyle[%d][%d][0] = %d, map_substyle[%d][%d][0] = %d\n",itype,jtype,map_substyle[itype][jtype][0],jtype,itype,map_substyle[jtype][itype][0]);
                
                if (!strcmp(keywords[substyle_ji],"sw")) {
                    int *map = (int *) styles[substyle_ji]->returnmap();
                    ietype= map[type[ii]];
                    
                    LAMMPS_NS::Pair::Param *params = (LAMMPS_NS::Pair::Param *) styles[substyle_ji]->returnparams();   // parameter set for an I-J-K interaction
                    int ***elem2param = (int ***) styles[substyle_ji]->returnelem2param();
                    
                    jetype=map[itype];
                    
                    ijparam = elem2param[ietype][jetype][jetype];
                    delr1[0] = coord[0] - x[ii][0];
                    delr1[1] = coord[1] - x[ii][1];
                    delr1[2] = coord[2] - x[ii][2];
                    rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
                    if (rsq1 > params[ijparam].cutsq) continue;
                    
                    for (int k = 0; k < nall; k++){
                        if (ii==k || k==i) continue;
                        if (mode == MOLECULE)
                            if (imolecule == molecule[ii]) continue;
                        
                        //printf("first possibility: nmap[%d][%d]= %d in for(int k = 0; k < nall; k++){}\n",jtype,ktype,nmap[jtype][ktype]);
                        ktype = type[k];
                        if (nmap[jtype][ktype] > 0) {
                            
                            int substyle_jk = map_substyle[jtype][ktype][0];
                            
                            if (!strcmp(keywords[substyle_jk],"sw")) {  // if yes, substyle_jk == sybstyle_ji
                                ketype = map[type[k]];
                                
                                delr2[0] = x[k][0] - x[ii][0];
                                delr2[1] = x[k][1] - x[ii][1];
                                delr2[2] = x[k][2] - x[ii][2];
                                
                                ikparam = elem2param[ietype][ketype][ketype];
                                ijkparam = elem2param[ietype][jetype][ketype];
                                
                                rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
                                if (rsq2 > params[ikparam].cutsq) continue;
                                styles[substyle_ji]->threebody(&params[ijparam],&params[ikparam],&params[ijkparam],rsq1,rsq2,delr1,delr2,fj,fk,eflag,tmp3body);
                                
                                total_energy+=tmp3body;
                            }
                        }
                    }
                }
            }
        }
        
        //printf("comm->me = %d: after first possibility loop\n",comm->me);
        
        //second possibility ii==i j!=i k!=i
        
        for (int j = 0; j < nall; j++){
            if (i==j) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[j]) continue;
            
            int **nmap = (int **) pair->returnnmap();
            
            //printf("second possibility: nmap[%d][%d]= %d in for(int k = 0; k < nall; k++){}\n",itype,jtype,nmap[itype][jtype]);
            jtype = type[j];
            if (nmap[itype][jtype] > 0) {
                
                int substyle_ij = map_substyle[itype][jtype][0];
                
                if (!strcmp(keywords[substyle_ij],"sw")) {
                    int *map = (int *) styles[substyle_ij]->returnmap();
                    jetype= map[type[j]];
                    ietype=map[itype];
                    
                    LAMMPS_NS::Pair::Param *params = (LAMMPS_NS::Pair::Param *) styles[substyle_ij]->returnparams();   // parameter set for an I-J-K interaction
                    int ***elem2param = (int ***) styles[substyle_ij]->returnelem2param();
                    
                    ijparam = elem2param[ietype][jetype][jetype];
                    delr1[0] = x[j][0] - coord[0];
                    delr1[1] = x[j][1] - coord[1];
                    delr1[2] = x[j][2] - coord[2];
                    rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
                    
                    if (rsq1 > params[ijparam].cutsq) continue;
                    for (int k = j; k < nall; k++) {
                        if (i==k || k == j ) continue;
                        
                        ktype = type[k];
                        
                        if (nmap[itype][ktype] > 0) {
                            
                            int substyle_ik = map_substyle[itype][ktype][0];
                            
                            if (!strcmp(keywords[substyle_ik],"sw")) {
                                ketype = map[type[k]];
                                
                                ikparam = elem2param[ietype][ketype][ketype];
                                ijkparam = elem2param[ietype][jetype][ketype];
                                delr2[0] = x[k][0] - coord[0];
                                delr2[1] = x[k][1] - coord[1];
                                delr2[2] = x[k][2] - coord[2];
                                rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
                                if (rsq2 > params[ikparam].cutsq) continue;
                                styles[substyle_ij]->threebody(&params[ijparam],&params[ikparam],&params[ijkparam],rsq1,rsq2,delr1,delr2,fj,fk,eflag,tmp3body);
                                total_energy+=tmp3body;
                            }
                        }
                    }
                }
            }
        }
        //printf("comm->me = %d: after second possibility loop\n",comm->me);
        ////////// TOTAL ENERGY/////
        //total_energy=threebodyeng+twobodyeng;
    } else if (strstr(pair_style,"sw")) {
        //if (comm->me == 0) printf("energy() for Stw_GCMC part\n");
        int *map = (int *) pair->returnmap();
        ietype=map[itype];
        
        LAMMPS_NS::Pair::Param *params = (LAMMPS_NS::Pair::Param *) pair->returnparams();   // parameter set for an I-J-K interaction
        int ***elem2param = (int ***) pair->returnelem2param();
        
        
        for(int j = 0; j < nall; j++){
            //if (comm->me == 0) printf("j = %d in two-body part in Stw_GCMC()\n",j);    // added by Jibao
            if (i == j) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[j]) continue;
            //if (comm->me == 0) printf("x[%d][0] = %f,coord = %d\n",j,x[j][0],coord);// added by Jibao
            //if (comm->me == 0) printf("coord[0] = %f\n",coord[0]);// added by Jibao
            delx = coord[0] - x[j][0];
            dely = coord[1] - x[j][1];
            delz = coord[2] - x[j][2];
            //if (comm->me == 0) printf("j = %d in two-body part in Stw_GCMC()\n",j);    // added by Jibao
            rsq = delx*delx + dely*dely + delz*delz;
            jetype=map[type[j]];
            ijparam = elem2param[ietype][jetype][jetype];
            
            if (rsq < params[ijparam].cutsq){
                pair->twobody(&params[ijparam],rsq,fpair,eflag,tmp2body);        twobodyeng+=tmp2body;
                //printf("# %d %d %d %d %d %f %e\n",ietype,type[j],ietype,jetype,ijparam,params[ijparam].epsilon,tmp2body);
            }
        }
        
        //first possibility ii!=i j==i k!=i or ii!=i; j!=i ; k==i
        jetype=map[itype];
        for(int ii = 0;ii < nall;ii++){
            if (ii==i) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[ii]) continue;
            ietype= map[type[ii]];
            ijparam = elem2param[ietype][jetype][jetype];
            delr1[0] = coord[0] - x[ii][0];
            delr1[1] = coord[1] - x[ii][1];
            delr1[2] = coord[2] - x[ii][2];
            rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
            if (rsq1 > params[ijparam].cutsq) continue;
            for (int k = 0; k < nall; k++){
                if (ii==k || k==i) continue;
                ketype = map[type[k]];
                delr2[0] = x[k][0] - x[ii][0];
                delr2[1] = x[k][1] - x[ii][1];
                delr2[2] = x[k][2] - x[ii][2];
                ikparam = elem2param[ietype][ketype][ketype];
                ijkparam = elem2param[ietype][jetype][ketype];
                rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
                if (rsq2 > params[ikparam].cutsq) continue;
                pair->threebody(&params[ijparam],&params[ikparam],&params[ijkparam],rsq1,rsq2,delr1,delr2,fj,fk,eflag,tmp3body);
                threebodyeng+=tmp3body;
            }
        }
        //second possibility ii==i j!=i k!=i
        ietype=map[itype];
        for (int j = 0; j < nall; j++){
            if (i==j) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[j]) continue;
            jetype = map[type[j]];
            ijparam = elem2param[ietype][jetype][jetype];
            delr1[0] = x[j][0] - coord[0];
            delr1[1] = x[j][1] - coord[1];
            delr1[2] = x[j][2] - coord[2];
            rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
            if (rsq1 > params[ijparam].cutsq) continue;
            for (int k = j; k < nall; k++) {
                if (i==k || k == j ) continue;
                ketype = map[type[k]];
                ikparam = elem2param[ietype][ketype][ketype];
                ijkparam = elem2param[ietype][jetype][ketype];
                delr2[0] = x[k][0] - coord[0];
                delr2[1] = x[k][1] - coord[1];
                delr2[2] = x[k][2] - coord[2];
                rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
                if (rsq2 > params[ikparam].cutsq) continue;
                pair->threebody(&params[ijparam],&params[ikparam],&params[ijkparam],rsq1,rsq2,delr1,delr2,fj,fk,eflag,tmp3body);
                threebodyeng+=tmp3body;
            }
        }
        ////////// TOTAL ENERGY/////
        total_energy=threebodyeng+twobodyeng;
        
        /*
         if (comm->me == 0) {
         printf("in energy(): total_energy= %e, twobodyeng= %e, threebodyeng= %e\n",total_energy,twobodyeng,threebodyeng);
         //printf("end of Stw_GCMC()\n");    // added by Jibao
         }
         */
    } else if (strcmp(pair_style,"hybrid/overlay") != 0) {
        for (int j = 0; j < nall; j++) { // from original lammps; commented by Jibao
            
            if (i == j) continue;
            if (mode == MOLECULE)
                if (imolecule == molecule[j]) continue;
            
            delx = coord[0] - x[j][0];
            dely = coord[1] - x[j][1];
            delz = coord[2] - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];
            
            if (rsq < cutsq[itype][jtype])
                total_energy +=
                pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);
        }   // from original lammps; commented by Jibao
    } else {
        error->all(FLERR,"fix gcmc/vp does not currently support hybrid/overlay pair style");
    }
    
    
    
    
    
    
    
    
    /*
     else if (strcmp(pair_style,"hybrid/overlay") != 0) {
     if (!strcmp(pair_style,"sw") || !strcmp(pair_style,"sw0") || !strcmp(pair_style,"sw/omp") || !strcmp(pair_style,"sw0/omp")) {
     
     } else {
     
     }
     } else {
     
     }
     */
    
    return total_energy;
}

/* ----------------------------------------------------------------------
   compute the energy of the given gas molecule in its current position
   sum across all procs that own atoms of the given molecule
------------------------------------------------------------------------- */

double FixGCMCVp::molecule_energy(tagint gas_molecule_id)
{
  double mol_energy = 0.0;
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->molecule[i] == gas_molecule_id) {
      mol_energy += energy(i,atom->type[i],gas_molecule_id,atom->x[i]);
    }

  double mol_energy_sum = 0.0;
  MPI_Allreduce(&mol_energy,&mol_energy_sum,1,MPI_DOUBLE,MPI_SUM,world);

  return mol_energy_sum;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixGCMCVp::energy_full()
{
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build();
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  if (modify->n_post_force) modify->post_force(vflag);
  if (modify->n_end_of_step) modify->end_of_step();

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixGCMCVp::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (ngas*random_equal->uniform());
  if ((iwhichglobal >= ngas_before) &&
      (iwhichglobal < ngas_before + ngas_local)) {
    int iwhichlocal = iwhichglobal - ngas_before;
    i = local_gas_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

tagint FixGCMCVp::pick_random_gas_molecule()
{
  int iwhichglobal = static_cast<int> (ngas*random_equal->uniform());
  tagint gas_molecule_id = 0;
  if ((iwhichglobal >= ngas_before) &&
      (iwhichglobal < ngas_before + ngas_local)) {
    int iwhichlocal = iwhichglobal - ngas_before;
    int i = local_gas_list[iwhichlocal];
    gas_molecule_id = atom->molecule[i];
  }

  tagint gas_molecule_id_all = 0;
  MPI_Allreduce(&gas_molecule_id,&gas_molecule_id_all,1,
                MPI_LMP_TAGINT,MPI_MAX,world);

  return gas_molecule_id_all;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVp::toggle_intramolecular(int i)
{
  if (atom->avec->bonds_allow)
    for (int m = 0; m < atom->num_bond[i]; m++)
      atom->bond_type[i][m] = -atom->bond_type[i][m];

  if (atom->avec->angles_allow)
    for (int m = 0; m < atom->num_angle[i]; m++)
      atom->angle_type[i][m] = -atom->angle_type[i][m];

  if (atom->avec->dihedrals_allow)
    for (int m = 0; m < atom->num_dihedral[i]; m++)
      atom->dihedral_type[i][m] = -atom->dihedral_type[i][m];

  if (atom->avec->impropers_allow)
    for (int m = 0; m < atom->num_improper[i]; m++)
      atom->improper_type[i][m] = -atom->improper_type[i][m];
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixGCMCVp::update_gas_atoms_list()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

  if (nlocal > gcmc_nmax) {
    memory->sfree(local_gas_list);
    gcmc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_gas_list");
  }

  ngas_local = 0;
    
    int *type = atom->type; // added by Jibao

  if (regionflag) {

    if (mode == MOLECULE) {

      tagint maxmol = 0;
      for (int i = 0; i < nlocal; i++) maxmol = MAX(maxmol,molecule[i]);
      tagint maxmol_all;
      MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
      double comx[maxmol_all];
      double comy[maxmol_all];
      double comz[maxmol_all];
      for (int imolecule = 0; imolecule < maxmol_all; imolecule++) {
        for (int i = 0; i < nlocal; i++) {
          if (molecule[i] == imolecule) {
            mask[i] |= molecule_group_bit;
          } else {
            mask[i] &= molecule_group_inversebit;
          }
        }
        double com[3];
        com[0] = com[1] = com[2] = 0.0;
        group->xcm(molecule_group,gas_mass,com);
        comx[imolecule] = com[0];
        comy[imolecule] = com[1];
        comz[imolecule] = com[2];
      }

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          if (domain->regions[iregion]->match(comx[molecule[i]],
             comy[molecule[i]],comz[molecule[i]]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }

    } else {
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == ngcmc_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == ngcmc_type)) { // Modified by Jibao
      //if (mask[i] & groupbit) {   // commented out by Jibao
        local_gas_list[ngas_local] = i;
        ngas_local++;
      }
    }
  }

  MPI_Allreduce(&ngas_local,&ngas,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ngas_local,&ngas_before,1,MPI_INT,MPI_SUM,world);
  ngas_before -= ngas_local;
}

/* ----------------------------------------------------------------------
  return acceptance ratios
------------------------------------------------------------------------- */

double FixGCMCVp::compute_vector(int n)
{
  if (n == 0) return ntranslation_attempts;
  if (n == 1) return ntranslation_successes;
  if (n == 2) return ninsertion_attempts;
  if (n == 3) return ninsertion_successes;
  if (n == 4) return ndeletion_attempts;
  if (n == 5) return ndeletion_successes;
  if (n == 6) return nrotation_attempts;
  if (n == 7) return nrotation_successes;
    if (n == 8) return energyout;   // added by Jibao
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixGCMCVp::memory_usage()
{
  double bytes = gcmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixGCMCVp::write_restart(FILE *fp)
{
  int n = 0;
  double list[4];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = next_reneighbor;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixGCMCVp::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = static_cast<int> (list[n++]);
}
