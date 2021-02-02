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
#include "fix_conc.h"
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
//#include "pair_sw0.h"        // added by Jibao
#include "pair_hybrid.h"        // added by Jibao


using namespace std;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};

/* ---------------------------------------------------------------------- */

FixConc::FixConc(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Esteban: Illegal fix gcmc command");

  if (atom->molecular == 2)
    error->all(FLERR,"Fix gcmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  //size_vector = 8; // commented out by Jibao
    size_vector = 1;    // added by Jibao according to Matias' 2012 lammps version, to output energyout
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args

  nevery = force->inumeric(FLERR,arg[3]);
  reactive_type = force->inumeric(FLERR,arg[4]);
  product_type = force->inumeric(FLERR,arg[5]);
  fraction = force->numeric(FLERR,arg[6]);
  seed = force->inumeric(FLERR,arg[7]);
  
 //Esteban: reactive_type, product_type, E, region, nreactions
    
    //molflag = 0; // variable in 2012 verion // Jibao
    pairflag = 0; // added by Jibao. from Matias
    //pressflag=0;    // added by Jibao. from Matias
    regionflag=0;   // added by Jibao. from Matias

  // read options from end of input line

  options(narg-8,&arg[8]);

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

  //ncycles = nexchanges + nmcmoves + nreactions; //Esteban: Agregar nreactions 

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  nchanges=0;
  
  
  //Esteban: nfreaction_attempts, nbreaction_attempts, nfreaction_successes, nbreaction_successes

    energyout=0.0;  // Matias

  gcmc_nmax = 0;
  local_gas_list = NULL;
  local_react_list = NULL; //Esteban
  local_prod_list = NULL;  //Esteban

   // if (comm->me == 0) printf("End of FixConc::FixConc()\n");
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixConc::options(int narg, char **arg)
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
     // if (comm->me == 0) printf("End of FixConc::options()\n");
}

/* ---------------------------------------------------------------------- */

FixConc::~FixConc()
{
   // printf("FixConc()");
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;
    
    //delete region_insert;   // from Matias; deleted by Jibao

  memory->destroy(local_gas_list);
  memory->destroy(local_react_list);
  memory->destroy(local_prod_list);
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
   // if (comm->me == 0) printf("End of FixConc::~FixConc()\n");
}

/* ---------------------------------------------------------------------- */

int FixConc::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixConc::init()
{
   // if (comm->me == 0) printf("Begins: FixConc::init()\n");

  triclinic = domain->triclinic;

  // decide whether to switch to the full_energy option
    
    //if (comm->me == 0) printf("Begins 2: FixConc::init()\n");
    
  if (full_flag) {
    char *id_pe = (char *) "thermo_pe";
    int ipe = modify->find_compute(id_pe);
    c_pe = modify->compute[ipe];
  }

  int *type = atom->type;

  if (mode == ATOM) {
    if (product_type <= 0 || product_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
    if (reactive_type <= 0 || reactive_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
  }
    
    //if (comm->me == 0) printf("Begins 3: FixConc::init()\n");

  // if mode == ATOM, warn if any deletable atom has a mol ID

  if ((mode == ATOM) && atom->molecule_flag) {
      /*
      if (comm->me == 0) {
          printf("Inside if ((mode == ATOM)): FixConc::init()\n");
          printf("atom->molecule_flag = %d\n",atom->molecule_flag);
      }
      */
    tagint *molecule = atom->molecule;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == reactive_type)
        if (molecule[i]) flag = 1;
      
      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 2: FixConc::init()\n");
      
    int flagall;
      
      //printf("comm->me = %d, flag = %d, flagall = %d, before MPI_ALLreduce()\n",comm->me,flag,flagall);
      
      //error->all(FLERR,"Kao 0 !!!!");    // added by Jibao
      
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
      
      //error->all(FLERR,"Kao 1 !!!!");    // added by Jibao
      
      //if (comm->me == 0) printf("Inside if ((mode == ATOM)) 3: FixConc::init()\n");
      //if (comm->me == 0) printf("flag = %d, flagall = %d, after MPI_ALLreduce()\n",flag,flagall);
      
      if (flagall && comm->me == 0) {
          //if (comm->me == 0) printf("Inside if if (flagall && comm->me == 0): FixConc::init()\n");    // added by Jibao
          //error->all(FLERR,"Kao 2 !!!!");    // added by Jibao
          error->all(FLERR,"Fix gcmc cannot exchange individual atoms belonging to a molecule");
      }
  }
    
    //if (comm->me == 0) printf("Begins 4: FixConc::init()\n");

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
    
    //if (comm->me == 0) printf("Begins 5: FixConc::init()\n");

  if (((mode == MOLECULE) && (atom->molecule_flag == 0)) ||
      ((mode == MOLECULE) && (!atom->tag_enable || !atom->map_style)))
    error->all(FLERR,
               "Fix gcmc molecule command requires that "
               "atoms have molecule attributes");

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one
    
    //if (comm->me == 0) printf("Begins 6: FixConc::init()\n");

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
    
    //if (comm->me == 0) printf("Begins 7: FixConc::init()\n");

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix gcmc in a 2d simulation");

  // create a new group for interaction exclusions
    
    //if (comm->me == 0) printf("Before 'create a new group for interaction exclusions': FixConc::init()\n");
    
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

  } else //gas_mass = atom->mass[ngcmc_type];

  //if (gas_mass <= 0.0)
  //  error->all(FLERR,"Illegal fix gcmc gas mass <= 0");

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

    //if (comm->me == 0) printf("Before 'compute beta, lambda, sigma, and the zz factor': FixConc::init()\n");
    
  // compute beta, lambda, sigma, and the zz factor

//  double lambda = sqrt(force->hplanck*force->hplanck/
//                       (2.0*MY_PI*gas_mass*force->mvv2e*
//                        force->boltz*reservoir_temperature));
//  sigma = sqrt(force->boltz*reservoir_temperature*tfac_insert/gas_mass/force->mvv2e);
    
//    if (!pressure_flag) // using pressure_flag to replace pressflag defined by Matias
//        zz = exp(beta*chemical_potential)/(pow(lambda,3.0));
//    else if(pressure_flag) {
//        zz = pressure/(force->boltz*4.184*reservoir_temperature*1000/6.02e23)/(1e30);   // from Matias; need to check the meaning
        //zz = pressure*fugacity_coeff*beta/force->nktv2p;    // from the original expression of 2015 version of lammps
//    }
    
    
  //zz = exp(beta*chemical_potential)/(pow(lambda,3.0)); // commented out by Jibao
  //if (pressure_flag) zz = pressure*fugacity_coeff*beta/force->nktv2p; // commented out by Jibao
    
    if (comm->me==0) { // added by Jibao; from Matias' version of lammps
        //printf("zz factor equals %e\n",zz); // added by Jibao; from Matias' version of lammps
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
    
    //printf("End of FixConc::init()\n");

}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixConc::pre_exchange()
{
  // just return if should not be called on this timestep
 //if (comm->me == 0) printf("Begin of FixConc::pre_exchange()\n");
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
  update_product_atoms_list();
  update_reactive_atoms_list();

  //printf("nprod=%d nreact=%d\n", nprod_local, nreact_local);

  if (full_flag) {
    error->all(FLERR,"gcmc/react does not allow full energy");

  } else {

    if (mode == MOLECULE) {
      error->all(FLERR,"gcmc/react does not allow mode MOLECULE");
    } else {
        if (nreact*1.0/nprod<fraction){
            attempt_label_change();
        }   
     }         
            //Esteban: Agregar la probabilidad de una freaction o breaction
      
      //domain->pbc();    // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
      //comm->exchange(); // added by Jibao; to prevent the error: ERROR on proc 0: Bond atoms 4205 4209 missing on proc 0 at step 285 (../neigh_bond.cpp:196)
  }
  next_reneighbor = update->ntimestep+nevery;
 //if (comm->me == 0) printf("End of FixConc::pre_exchange()\n");
}

void FixConc::attempt_label_change()
{
  nchanges+=1;

  int success = 0;
  
  int i = pick_random_product_atom();

  if (i >= 0){

    int *type = atom->type;

    type[i] = reactive_type;
    success = 1;
    
}

    
    int success_all = 0;
    //printf("antes del MPI\n");
    MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);
    //printf("despues del MPI\n");
    if (success_all) {
        update_gas_atoms_list();
        update_reactive_atoms_list();
        update_product_atoms_list();
        
        atom->nghost = 0;
        comm->borders();

    }
    //printf("freaction end");
}


/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixConc::pick_random_reactive_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nreact*random_equal->uniform());
  if ((iwhichglobal >= nreact_before) &&
      (iwhichglobal < nreact_before + nreact_local)) {
    int iwhichlocal = iwhichglobal - nreact_before;
    i = local_react_list[iwhichlocal];
  }
  //printf("iwhichglobal = %i, ngas_before = %i, ngas_local = %i\n", iwhichglobal, ngas_before, ngas_local);

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixConc::pick_random_product_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nprod*random_equal->uniform());
  if ((iwhichglobal >= nprod_before) &&
      (iwhichglobal < nprod_before + nprod_local)) {
    int iwhichlocal = iwhichglobal - nprod_before;
    i = local_prod_list[iwhichlocal];
  }
  //printf("iwhichglobal = %i, ngas_before = %i, ngas_local = %i\n", iwhichglobal, ngas_before, ngas_local);
  return i;
}


/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

tagint FixConc::pick_random_gas_molecule()
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

void FixConc::toggle_intramolecular(int i)
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
//Esteban: asegurarse de que actualize correctamente luego de una reaccion

void FixConc::update_gas_atoms_list()
{
//printf("Begin of FixConc::update_gas_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

  if (nlocal > gcmc_nmax) {
    memory->sfree(local_gas_list);
    gcmc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_gas_list");
    memory->sfree(local_react_list);
    gcmc_nmax = atom->nmax;
    local_react_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_react_list");
    memory->sfree(local_prod_list);
    gcmc_nmax = atom->nmax;
    local_prod_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_prod_list");
  }

  ngas_local = 0;
 //printf("End of FixConc::update_gas_atoms_list()\n");

}

/* ----------------------------------------------------------------------
   update the list of reactive atoms
------------------------------------------------------------------------- */

void FixConc::update_reactive_atoms_list()
{
 //if (comm->me == 0) printf("Begin of FixConc::update_reactive_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;
  
//    if (nlocal > gcmc_nmax) {
//    printf("Hasta aca\n");
//    memory->sfree(local_react_list);
//    gcmc_nmax = atom->nmax;
//    local_react_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
//     "GCMC:local_react_list");
//  }

  nreact_local = 0;
    
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

    } else { //Esteban: modificado para trabajar con el numero de reactivos
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == reactive_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_react_list[nreact_local] = i;
            nreact_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == reactive_type)) { // Modified by Jibao
        //if (type[i] == reactive_type) {
      //if (mask[i] & groupbit) {   // commented out by Jibao
        local_react_list[nreact_local] = i;
        nreact_local++;
      }
    }
  }

  MPI_Allreduce(&nreact_local,&nreact,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nreact_local,&nreact_before,1,MPI_INT,MPI_SUM,world);
  nreact_before -= nreact_local;
  //printf("proc=%i, nlocal=%i, nreact_local=%i, nreact_before=%i\n", comm->me, nlocal, nreact_local, nreact_before);
}

/* ----------------------------------------------------------------------
   update the list of product atoms
------------------------------------------------------------------------- */

void FixConc::update_product_atoms_list()
{
 //if (comm->me == 0) printf("Begin of FixConc::update_product_atoms_list()\n");
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

//  if (nlocal > gcmc_nmax) {
//    memory->sfree(local_prod_list);
//    gcmc_nmax = atom->nmax;
//    local_prod_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
//     "GCMC:local_prod_list");
//  }

  nprod_local = 0;
    
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

    } else { //Esteban: modificado para trabajar con el numero de productos
      for (int i = 0; i < nlocal; i++) {
          if ((mask[i] & groupbit) && (type[i] == product_type)) {  // Modified by Jibao
        //if (mask[i] & groupbit) { // commented out by Jibao
          if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_prod_list[nprod_local] = i;
            nprod_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == product_type)) { // Modified by Jibao
        //if (type[i] == reactive_type) {
      //if (mask[i] & groupbit) {   // commented out by Jibao
        local_prod_list[nprod_local] = i;
        nprod_local++;
      }
    }
  }

  MPI_Allreduce(&nprod_local,&nprod,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nprod_local,&nprod_before,1,MPI_INT,MPI_SUM,world);
  nprod_before -= nprod_local;
  //printf("proc=%i, nlocal=%i, nprod_local=%i, nprod_before=%i\n", comm->me, nlocal, nprod_local, nprod_before);
}

/* ----------------------------------------------------------------------
  return acceptance ratios
------------------------------------------------------------------------- */

double FixConc::compute_vector(int n)
{
  if (n == 0) return nchanges;
  
  return 0.0;
}
//Esteban:Agregar los nuevos eventos

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixConc::memory_usage()
{
  double bytes = gcmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixConc::write_restart(FILE *fp)
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

void FixConc::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = static_cast<int> (list[n++]);
}
