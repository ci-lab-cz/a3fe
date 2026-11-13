Batch workflow (a3fe-abfe)
==========================

``a3fe-abfe`` is a command line tool which prepares, runs, tracks and resumes absolute binding free energy calculations
for many ligands on a PBS (or SLURM) cluster. It is installed with a3fe.

.. note::

    The tool has been used with PBS (OpenPBS) on CPU nodes with the GROMACS engine. ``scheduler: slurm`` has not been tested yet.

How it works
------------

For every ligand one small **driver job** is submitted to the scheduler. The driver

1. parameterises the protein (pdb4amber + tleap, ff14SB) and the ligand (OpenFF 2.0.0), assembles the bound and free
   systems and renames the ligand residue to LIG;
2. runs the a3fe setup (solvation, minimisation, pre-equilibration, ensemble equilibration) - each step is a job
   submitted by the driver;
3. submits one job per production window and waits for them, repeating windows which failed;
4. analyses the ligand and writes ``lig_NAME/output/overall_stats.dat``;
5. starts the next waiting ligand(s).

Nothing has to stay running on the login node, several ligands are processed at the same time (``max_active_ligands``),
and the state of each ligand is kept in ``lig_NAME/.abfe/state.json``, so the progress can be checked and interrupted
ligands can be resumed at any time.

Quick start
-----------

.. code-block:: bash

    conda activate a3fe            # the environment is re-activated inside all jobs

    # 1. Create a batch directory with one sub-directory per ligand
    a3fe-abfe -d batch_01 init --protein protein.pdb --ligands docked_ligands.sdf

    # 2. Check the settings (queue, cores, protocol)
    cd batch_01 && nano abfe.yaml

    # 3. Start
    a3fe-abfe submit

    # 4. Follow the progress
    a3fe-abfe status

    # 5. Collect the results
    a3fe-abfe collect              # -> abfe_results.tsv

All commands are run in the batch directory (or with ``-d <batch dir>``).

The conda environment which is active when ``a3fe-abfe submit`` is called is activated in all jobs. If no conda
environment is active, set up the environment with ``setup_commands`` in ``abfe.yaml``.

Input
-----

``init`` creates ``lig_<NAME>/input/`` for each molecule of the SDF file (the name is taken from the title line),
containing

.. code-block:: text

    ligand.sdf      one ligand: docked pose, all hydrogens, correct protonation
    protein.pdb     prepared protein (protonation / HIS states assigned)

and copies the settings file ``abfe.yaml`` into the batch directory. The directories can also be made by hand. Optional
files in ``input/``:

- ``water.parm7`` + ``water.rst7``: crystallographic waters, added to the bound system
  (``init --water water.parm7 water.rst7`` copies them for every ligand);
- ``protein.parm7`` + ``protein.rst7``: an already parameterised protein (then ``protein.pdb`` is not needed);
- ``bound_param.prm7/.rst7`` + ``free_param.prm7/.rst7``: fully parameterised systems - the preparation is then skipped.

Ligands can be added to a batch at any time: create the directories (e.g. run ``init`` again with another SDF file -
existing ligands are skipped) and run ``a3fe-abfe submit``.

Settings (abfe.yaml)
--------------------

The file is commented. The most important options:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Meaning
   * - ``scheduler``
     - ``pbs`` or ``slurm``
   * - ``engine``
     - ``gromacs`` (CPU or GPU nodes) or ``somd`` (GPU nodes only)
   * - ``max_active_ligands``
     - ligands processed at the same time
   * - ``submit_delay``
     - seconds between the submissions of the production jobs of a ligand
   * - ``setup_commands``
     - commands run at the start of each job (e.g. ``module load ...``)
   * - ``driver``
     - queue, walltime, cores and memory of the driver job
   * - ``simulation``
     - queue, walltime, cores, GPUs and memory of each simulation job
   * - ``protocol``
     - replicates, run time, equilibration time, set-up times, lambda schedule, engine options

``queue`` is the PBS queue or the SLURM partition. ``walltime: null`` submits the jobs without a time limit, so that the
default of the queue applies. ``extra_options`` are passed to the scheduler as they are, e.g. ``{A: my_account}`` for
PBS (``#PBS -A my_account``) or ``{account: my_account}`` for SLURM (``#SBATCH --account=my_account``).

CPU nodes: ``engine: gromacs``, ``simulation.ngpus: 0`` and ``simulation.ncpus`` set to the cores of one simulation.
GROMACS uses all the cores given to the job. GPU nodes: ``simulation.ngpus: 1``.

Always request memory for the jobs (``mem``): some PBS installations give jobs without a memory request very little.

Changed resources are used by the jobs submitted from then on, also for ligands which are already running. Changes of
the protocol only affect ligands whose setup has not been done yet.

Commands
--------

.. code-block:: text

    a3fe-abfe submit                  start / resume all ligands which are new or were interrupted
    a3fe-abfe submit lig_A lig_B      (re)start these ligands, also if they failed or were cancelled
    a3fe-abfe submit --retry          also restart all failed and cancelled ligands
    a3fe-abfe status                  table of all ligands
    a3fe-abfe status --state failed   only ligands in the given state(s)
    a3fe-abfe status lig_A            details: stages, jobs, error, end of the driver log
    a3fe-abfe cancel lig_A            stop a ligand and delete its jobs  (--all: every ligand)
    a3fe-abfe reset lig_A             delete all output of a ligand, keeping its input
    a3fe-abfe collect                 write abfe_results.tsv (id, dG, CI95, dG of each run)

Selecting ligands
~~~~~~~~~~~~~~~~~

A ligand is a sub-directory of the batch directory whose name starts with ``lig_`` and which contains an ``input/``
directory. Everything else in the batch directory is ignored.

Ligands are given to ``submit``, ``status``, ``cancel`` and ``reset`` by the name of their directory, not by a path.
The ``lig_`` prefix can be omitted and a trailing ``/`` is accepted, so ``lig_A``, ``A`` and ``lig_A/`` are the same
ligand.

Without ligand names ``submit`` goes through all ligands of the batch:

- ``new`` and ``interrupted`` ligands are started / resumed;
- ``done``, ``queued`` and ``running`` ligands are skipped;
- ``failed`` and ``cancelled`` ligands are skipped, unless ``--retry`` is given;
- at most ``max_active_ligands`` are active at the same time (``--max-active N`` overrides it for this call), the
  others are started automatically as running ligands finish.

Ligands which are named explicitly are also restarted if they are ``failed`` or ``cancelled``.

States
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - State
     - Meaning
   * - ``new``
     - not started yet
   * - ``queued``
     - driver job waiting in the queue
   * - ``running``
     - driver job running; the stage is ``prepare``, ``setup``, ``production`` or ``analysis``
   * - ``done``
     - finished, result in ``output/overall_stats.dat``
   * - ``failed``
     - an error occurred (shown by ``status``, details with ``status <ligand>``)
   * - ``interrupted``
     - the driver job disappeared (walltime, node failure, deleted)
   * - ``cancelled``
     - stopped with ``a3fe-abfe cancel``

In the status table ``SIMS`` is the number of finished / all production simulations, and ``JOBS R/Q`` the number of
running / waiting simulation jobs of the ligand.

Resuming
~~~~~~~~

``a3fe-abfe submit`` can be repeated at any time; it never starts a ligand twice.

- ``interrupted`` ligands are resumed where they stopped: finished preparation steps and finished production windows
  are kept, only unfinished windows are run again. A ligand is resumed automatically at most ``max_attempts`` times.
- ``failed`` and ``cancelled`` ligands are only restarted when named explicitly or with ``--retry`` (they also continue
  from where they stopped).
- To start a ligand from scratch use ``a3fe-abfe reset <ligand>`` and then ``submit``.

Simulation jobs left over from an interrupted driver are deleted when the ligand is resumed.

What is handled automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A production simulation which failed, or whose job disappeared without any output (e.g. a problem of its node), is
  repeated up to ``max_sim_retries`` times. A failed set-up step is repeated the same number of times.
- If the scheduler can temporarily not be reached, its commands are repeated for about 10 minutes.
- If the PBS server refuses a submission with ``Job with requested ID already exists`` (job ids still used by old jobs
  in the job history of the server), the submission is repeated until a free id is reached.

Files
-----

.. code-block:: text

    batch_01/
      abfe.yaml
      abfe_results.tsv              written by "collect"
      lig_<NAME>/
        .abfe/state.json            state of the ligand
        .abfe/driver.log            log of the driver job(s) - look here first if something fails
        .abfe/driver.sh             the submitted driver script
        input/                      input, parameters and set-up files (solvated, minimised, ...)
        bound/, free/               legs: ensemble_equilibration_N/ and <stage>/output/lambda_X/run_NN/
        output/overall_stats.dat    final dG, written last

The jobs of a ligand are named ``drv.lig_<NAME>`` (driver) and ``sim.lig_<NAME>`` (simulations) in the queue. With the
default GROMACS protocol a ligand needs 12 set-up jobs and 48 lambda windows x ``ensemble_size`` production jobs (240
for 5 replicates).

Requirements
------------

- Compute nodes must be allowed to submit jobs (the driver job submits the simulation jobs).
- The batch directory has to be on a file system shared by all nodes.
- ``pdb4amber`` and ``tleap`` (AmberTools) are needed to parameterise the protein.
