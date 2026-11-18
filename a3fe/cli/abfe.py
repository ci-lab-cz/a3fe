"""
a3fe-abfe - prepare, run, track and resume a3fe ABFE calculations for many
ligands on a PBS (or SLURM) cluster.

Each ligand lives in its own directory (lig_<NAME>/input/{ligand.sdf,protein.pdb})
inside a batch directory which also holds the settings (abfe.yaml).

For every ligand one small "driver" job is submitted to the scheduler. The
driver parameterises the inputs, runs the a3fe setup, submits and tracks the
production simulations, analyses them, and finally submits the next waiting
ligand. Nothing has to stay running on the login node.

    a3fe-abfe init --protein protein.pdb --ligands ligands.sdf
    a3fe-abfe submit              start / resume everything which is not done
    a3fe-abfe status              overview of all ligands
    a3fe-abfe status lig_X        details of one ligand
    a3fe-abfe cancel lig_X        stop a ligand (or --all)
    a3fe-abfe hold                pause the batch to free the nodes
    a3fe-abfe release             continue a paused batch
    a3fe-abfe collect             table of the results

See the "Batch workflow" page of the documentation for details.
"""

import argparse
import datetime
import fcntl
import getpass
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

PROGRAM = "a3fe-abfe"
# The module is run with "python -m" inside the driver jobs
MODULE = "a3fe.cli.abfe"
CONFIG_NAME = "abfe.yaml"
# The settings file copied into new batch directories
CONFIG_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "abfe_template.yaml"
)
STATE_DIR = ".abfe"
# Present in the batch directory while the batch is paused with 'hold'
HOLD_NAME = ".abfe.hold"
HOLD_CHECK_INTERVAL = 60  # s
LIG_PREFIX = "lig_"

DEFAULTS = {
    "scheduler": "pbs",  # pbs | slurm
    "engine": "gromacs",  # gromacs (CPU or GPU) | somd (GPU only)
    "max_active_ligands": 10,
    "max_attempts": 3,
    "setup_commands": [],
    "submit_delay": 1.0,
    "driver": {
        "queue": None,
        "walltime": "72:00:00",
        "ncpus": 2,
        "mem": "16gb",
        "extra_options": {},
    },
    "simulation": {
        "queue": None,
        "walltime": "04:00:00",
        "ncpus": 8,
        "ngpus": 0,
        "mem": "8gb",
        "extra_options": {},
    },
    "protocol": {
        "ensemble_size": 5,
        "runtime": 0.1,
        "equilibration_time": 0.02,
        "sysprep": {},
        "lambda_values": None,
        "engine_options": {},
        "max_sim_retries": 2,
        "lighten": False,
    },
}

# A PBS server with job history enabled can hand out job ids which are still
# used by old jobs in its history. qsub then fails, and every failed attempt
# moves the server on to the next id, so the submission is simply repeated.
PBS_ID_COLLISION = "Job with requested ID already exists"
PBS_ID_RETRIES = 3000
PBS_ID_RETRY_WAIT = 1  # s

# Messages of the scheduler commands when the server can temporarily not be
# reached (e.g. it is busy, or too many commands are run at the same time on
# one node). The command is then repeated.
SERVER_ERRORS = (
    "cannot connect to server",
    "pbs_iff",
    "Unable to authenticate",
    "Connection refused",
    "Connection timed out",
    "Socket timed out",
    "Unable to contact slurm controller",
)
SERVER_RETRIES = 60
SERVER_RETRY_WAIT = 10  # s

# States of a ligand
NEW, QUEUED, RUNNING, DONE, FAILED = "new", "queued", "running", "done", "failed"
INTERRUPTED, CANCELLED = "interrupted", "cancelled"


class Interrupted(Exception):
    """Raised when the driver is asked to stop (e.g. walltime reached)."""


# --------------------------------------------------------------------------- #
# Configuration and ligand state
# --------------------------------------------------------------------------- #


def merge(defaults, values):
    """Recursively overlay values on the defaults."""
    result = dict(defaults)
    for key, value in (values or {}).items():
        if key not in defaults:
            sys.exit(f"Unknown option '{key}' in {CONFIG_NAME}")
        if isinstance(defaults[key], dict) and key not in (
            "extra_options",
            "sysprep",
            "engine_options",
        ):
            result[key] = merge(defaults[key], value)
        else:
            result[key] = value
    return result


def load_config(batch_dir):
    path = os.path.join(batch_dir, CONFIG_NAME)
    if not os.path.isfile(path):
        sys.exit(
            f"No {CONFIG_NAME} in {batch_dir}. Create the batch directory with "
            f"'{PROGRAM} init', or use -d <batch dir>."
        )
    with open(path) as f:
        cfg = merge(DEFAULTS, yaml.safe_load(f))
    if cfg["scheduler"] not in ("pbs", "slurm"):
        sys.exit("scheduler must be 'pbs' or 'slurm'")
    if cfg["engine"] not in ("gromacs", "somd"):
        sys.exit("engine must be 'gromacs' or 'somd'")
    if cfg["engine"] == "somd" and not cfg["simulation"]["ngpus"]:
        sys.exit("The somd engine only runs on GPUs: set simulation.ngpus to 1")
    return cfg


def list_ligands(batch_dir):
    return sorted(
        d
        for d in os.listdir(batch_dir)
        if d.startswith(LIG_PREFIX)
        and os.path.isdir(os.path.join(batch_dir, d, "input"))
    )


def select_ligands(batch_dir, names):
    """Return the named ligands (the 'lig_' prefix is optional), or all."""
    ligands = list_ligands(batch_dir)
    if not names:
        return ligands
    selected = []
    for name in names:
        name = name.rstrip("/")
        name = name if name in ligands else LIG_PREFIX + name
        if name not in ligands:
            sys.exit(f"No such ligand: {name}")
        selected.append(name)
    return selected


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def state_path(batch_dir, lig):
    return os.path.join(batch_dir, lig, STATE_DIR, "state.json")


def read_state(batch_dir, lig):
    try:
        with open(state_path(batch_dir, lig)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": NEW}


def write_state(batch_dir, lig, **changes):
    state = read_state(batch_dir, lig)
    state.update(changes)
    state["updated"] = now()
    path = state_path(batch_dir, lig)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)
    return state


def effective_state(state, active_jobs):
    """The state of a ligand, taking into account whether its driver job
    is still alive: a queued / running ligand without a job was interrupted."""
    name = state.get("state", NEW)
    if name in (QUEUED, RUNNING) and str(state.get("job_id")) not in active_jobs:
        return INTERRUPTED
    return name


def read_result(batch_dir, lig):
    """Return (dG, CI95, [dG of each run]) or None."""
    try:
        with open(os.path.join(batch_dir, lig, "output", "overall_stats.dat")) as f:
            text = f.read()
    except FileNotFoundError:
        return None
    mean = re.search(r"Mean free energy:\s*(-?[\d.]+)\s*\+\s*/-\s*([\d.]+)", text)
    if not mean:
        return None
    runs = re.findall(r"from run \d+:\s*(-?[\d.]+)", text)
    return mean.group(1), mean.group(2), runs


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def batch_tag(batch_dir):
    """A short identifier of the batch directory. It is part of the job names,
    so that the jobs of ligands with the same name in different batches are
    not mixed up."""
    return hashlib.sha1(os.path.realpath(batch_dir).encode()).hexdigest()[:6]


def driver_job_name(batch_dir, lig):
    return f"drv.{lig}.{batch_tag(batch_dir)}"


def sim_job_name(batch_dir, lig):
    return f"sim.{lig}.{batch_tag(batch_dir)}"


def batch_jobs(batch_dir, active_jobs):
    """Return the ids of the waiting simulation jobs, the waiting driver jobs
    and the running simulation jobs of the batch."""
    ligands = list_ligands(batch_dir)
    sim_names = {sim_job_name(batch_dir, lig) for lig in ligands}
    drv_names = {driver_job_name(batch_dir, lig) for lig in ligands}
    waiting_sims = [j for j, (n, s) in active_jobs.items() if n in sim_names and s != "R"]  # fmt: skip
    waiting_drvs = [j for j, (n, s) in active_jobs.items() if n in drv_names and s != "R"]  # fmt: skip
    running_sims = [j for j, (n, s) in active_jobs.items() if n in sim_names and s == "R"]  # fmt: skip
    return waiting_sims, waiting_drvs, running_sims


def hold_path(batch_dir):
    return os.path.join(batch_dir, HOLD_NAME)


def is_held(batch_dir):
    return os.path.isfile(hold_path(batch_dir))


def read_hold(batch_dir):
    try:
        with open(hold_path(batch_dir)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def wait_while_held(batch_dir):
    """Block while the batch is paused. Returns True if it had to wait."""
    if not is_held(batch_dir):
        return False
    print(f"[{now()}] The batch is on hold - waiting for 'release'", flush=True)
    while is_held(batch_dir):
        time.sleep(HOLD_CHECK_INTERVAL)
    print(f"[{now()}] The batch was released - continuing", flush=True)
    return True


def conda_setup_commands():
    """Commands which activate the currently active conda environment in a job."""
    prefix, exe = os.environ.get("CONDA_PREFIX"), os.environ.get("CONDA_EXE")
    if not prefix or not exe:
        return []
    conda_sh = os.path.join(
        os.path.dirname(os.path.dirname(exe)), "etc/profile.d/conda.sh"
    )
    if not os.path.isfile(conda_sh):
        return []
    return [f"source {shlex.quote(conda_sh)}", f"conda activate {shlex.quote(prefix)}"]


class Scheduler:
    """The few scheduler operations needed outside of a3fe."""

    def __init__(self, cfg):
        self.kind = cfg["scheduler"]
        self.cfg = cfg

    def active_jobs(self):
        """Return {job id: (job name, 'R' | 'Q')} for the user's unfinished jobs."""
        jobs = {}
        user = getpass.getuser()
        if self.kind == "pbs":
            out = self._run(["qstat", "-f", "-w"])
            for block in re.split(r"^Job Id: ", out, flags=re.M)[1:]:
                fields = dict(re.findall(r"^\s+(\S+) = (.*)$", block, flags=re.M))
                if fields.get("Job_Owner", "").split("@")[0] != user:
                    continue
                state = fields.get("job_state", "")
                if state in "EFXM":  # exiting / finished / moved
                    continue
                job_id = block.split()[0].split(".")[0]
                jobs[job_id] = (
                    fields.get("Job_Name", ""),
                    "R" if state == "R" else "Q",
                )
        else:
            out = self._run(
                ["squeue", "-h", "-u", user, "-t", "R,PD,S,CG", "-o", "%i|%j|%t"]
            )
            for line in out.splitlines():
                job_id, name, state = line.strip().split("|")
                jobs[job_id] = (name, "R" if state == "R" else "Q")
        return jobs

    @staticmethod
    def _run(cmd):
        for attempt in range(SERVER_RETRIES + 1):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            unreachable = any(error in result.stderr for error in SERVER_ERRORS)
            if not unreachable or attempt == SERVER_RETRIES:
                break
            time.sleep(SERVER_RETRY_WAIT)
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")

    def cancel(self, job_ids):
        self._apply(["qdel"] if self.kind == "pbs" else ["scancel"], job_ids)

    def hold(self, job_ids):
        """Keep waiting jobs from starting."""
        self._apply(["qhold"] if self.kind == "pbs" else ["scontrol", "hold"], job_ids)

    def release(self, job_ids):
        """Let held jobs start again (harmless for jobs which are not held)."""
        self._apply(
            ["qrls"] if self.kind == "pbs" else ["scontrol", "release"], job_ids
        )

    @staticmethod
    def _apply(cmd, job_ids):
        job_ids = [str(j) for j in job_ids]
        for i in range(0, len(job_ids), 200):
            chunk = job_ids[i : i + 200]
            # scontrol takes a comma separated list of job ids
            chunk = [",".join(chunk)] if cmd[0] == "scontrol" else chunk
            subprocess.run(cmd + chunk, capture_output=True)

    def submit_driver(self, batch_dir, lig):
        """Write and submit the driver job of a ligand. Returns the job id."""
        res = self.cfg["driver"]
        log = os.path.join(batch_dir, lig, STATE_DIR, "driver.log")
        script_path = os.path.join(batch_dir, lig, STATE_DIR, "driver.sh")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        lines = ["#!/bin/bash"]
        if self.kind == "pbs":
            select = f"1:ncpus={res['ncpus']}" + (
                f":mem={res['mem']}" if res["mem"] else ""
            )
            lines += [f"#PBS -N {driver_job_name(batch_dir, lig)}"]
            lines += [f"#PBS -q {res['queue']}"] if res["queue"] else []
            lines += [
                f"#PBS -l select={select}",
            ]
            lines += [f"#PBS -l walltime={res['walltime']}"] if res["walltime"] else []
            lines += [
                "#PBS -j oe",
                "#PBS -R oe",
            ]
            lines += [f"#PBS -{k} {v}" for k, v in res["extra_options"].items()]
            lines += [f"exec >> {shlex.quote(log)} 2>&1"]
        else:
            lines += [f"#SBATCH --job-name={driver_job_name(batch_dir, lig)}"]
            lines += [f"#SBATCH --partition={res['queue']}"] if res["queue"] else []
            lines += [f"#SBATCH --time={res['walltime']}"] if res["walltime"] else []
            lines += [
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                f"#SBATCH --cpus-per-task={res['ncpus']}",
                f"#SBATCH --output={log}",
                "#SBATCH --open-mode=append",
            ]
            lines += [f"#SBATCH --mem={res['mem']}"] if res["mem"] else []
            lines += [f"#SBATCH --{k}={v}" for k, v in res["extra_options"].items()]
        python = shlex.quote(sys.executable)
        call = f"{python} -m {MODULE} -d {shlex.quote(batch_dir)}"
        lines += [
            "",
            f"cd {shlex.quote(batch_dir)} || exit 1",
            *self.cfg["setup_commands"],
            *conda_setup_commands(),
            "",
            f"{call} run {shlex.quote(lig)}",
            "# Start the next waiting ligand(s)",
            f"{call} submit --from-driver",
        ]
        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        cmd = (
            ["qsub", script_path]
            if self.kind == "pbs"
            else ["sbatch", "--parsable", script_path]
        )
        for _ in range(PBS_ID_RETRIES):
            try:
                out = self._run(cmd).strip()
                break
            except RuntimeError as e:
                if PBS_ID_COLLISION not in str(e):
                    raise
                time.sleep(PBS_ID_RETRY_WAIT)
        else:
            raise RuntimeError(
                f"qsub failed {PBS_ID_RETRIES} times: {PBS_ID_COLLISION}"
            )
        return out.splitlines()[-1].split(".")[0].split(";")[0]


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


def cmd_init(args):
    """Create one lig_<NAME>/input directory for each molecule of an SDF file."""
    batch_dir = args.dir
    os.makedirs(batch_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(batch_dir, CONFIG_NAME)):
        shutil.copy(CONFIG_TEMPLATE, os.path.join(batch_dir, CONFIG_NAME))
        print(f"Copied {CONFIG_NAME} to {batch_dir} - please check the settings")
    with open(args.ligands) as f:
        records = [r.lstrip("\n") for r in f.read().split("$$$$") if r.strip()]
    n_new = 0
    for i, record in enumerate(records, 1):
        title = record.splitlines()[0].strip() or f"{i:04d}"
        name = re.sub(r"[^A-Za-z0-9_.+-]", "_", title)
        input_dir = os.path.join(batch_dir, LIG_PREFIX + name, "input")
        if os.path.isdir(input_dir):
            print(f"  exists, skipped: {LIG_PREFIX}{name}")
            continue
        os.makedirs(input_dir)
        with open(os.path.join(input_dir, "ligand.sdf"), "w") as f:
            f.write(record.rstrip("\n") + "\n$$$$\n")
        shutil.copy(args.protein, os.path.join(input_dir, "protein.pdb"))
        for extra in args.water or []:
            shutil.copy(extra, input_dir)
        n_new += 1
    print(f"Created {n_new} ligand directories in {batch_dir}")


# --------------------------------------------------------------------------- #
# submit / cancel / reset
# --------------------------------------------------------------------------- #


class BatchLock:
    """Make sure that only one process submits drivers at a time."""

    def __init__(self, batch_dir):
        self.path = os.path.join(batch_dir, ".abfe.lock")

    def __enter__(self):
        self.file = open(self.path, "w")
        fcntl.flock(self.file, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def cmd_submit(args):
    batch_dir, cfg = args.dir, load_config(args.dir)
    sched = Scheduler(cfg)
    explicit = bool(args.ligands)
    ligands = select_ligands(batch_dir, args.ligands)
    max_active = args.max_active or cfg["max_active_ligands"]
    if is_held(batch_dir):
        # No new drivers while the batch is paused; 'release' starts them
        if args.from_driver:
            print(f"[{now()}] the batch is on hold - no ligands submitted")
            return
        sys.exit(f"The batch is on hold - continue it with '{PROGRAM} release'.")
    submitted, n_active, waiting, skipped = submit_ligands(
        batch_dir, cfg, sched, ligands, max_active, explicit or args.retry
    )
    if args.from_driver:
        print(f"[{now()}] submitted next ligands: {' '.join(submitted) or 'none'}")
        return
    print(
        f"Submitted {len(submitted)} ligand(s); {n_active} active (limit {max_active})."
    )
    if waiting > 0:
        print(
            f"{waiting} more will be started automatically as running ligands finish."
        )
    if skipped:
        print(
            f"{len(skipped)} interrupted ligand(s) were not resumed because they already "
            f"stopped {cfg['max_attempts']} times - check them with 'status <ligand>' and "
            "resume with 'submit <ligand>' or 'submit --retry'."
        )


def submit_ligands(batch_dir, cfg, sched, ligands, max_active, retry, resume=()):
    """Submit the drivers of the ligands which are due, up to max_active active
    ligands. Ligands in 'resume' are also restarted if they failed.
    Returns (submitted ligands, number of active ligands, number of ligands
    still waiting, ligands which were interrupted too often)."""
    with BatchLock(batch_dir):
        active_jobs = sched.active_jobs()
        states = {lig: read_state(batch_dir, lig) for lig in list_ligands(batch_dir)}
        eff = {lig: effective_state(s, active_jobs) for lig, s in states.items()}
        n_active = sum(e in (QUEUED, RUNNING) for e in eff.values())
        submitted, skipped = [], []
        for lig in ligands:
            state, e = states[lig], eff[lig]
            if e in (DONE, QUEUED, RUNNING):
                continue
            again = retry or (lig in resume and e != CANCELLED)
            if e in (FAILED, CANCELLED) and not again:
                continue
            if e == INTERRUPTED and not again:
                # Do not resume a ligand whose driver keeps dying forever
                if state.get("attempts", 0) >= cfg["max_attempts"]:
                    skipped.append(lig)
                    continue
            if n_active >= max_active:
                break
            attempts = 1 if again or e == NEW else state.get("attempts", 0) + 1
            job_id = sched.submit_driver(batch_dir, lig)
            write_state(
                batch_dir, lig, state=QUEUED, job_id=job_id, attempts=attempts,
                error=None, submitted=now(),
            )  # fmt: skip
            submitted.append(lig)
            n_active += 1
    waiting = (
        sum(e in (NEW, INTERRUPTED) for e in eff.values())
        - len(submitted)
        - len(skipped)
    )
    return submitted, n_active, waiting, skipped


def kill_ligand_jobs(sched, batch_dir, lig, driver=True):
    """Cancel the simulation jobs (and optionally the driver) of a ligand."""
    names = [sim_job_name(batch_dir, lig)]
    names += [driver_job_name(batch_dir, lig)] if driver else []
    job_ids = [j for j, (name, _) in sched.active_jobs().items() if name in names]
    # Never cancel the job we are running in
    own = (os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "").split(
        "."
    )[0]
    job_ids = [j for j in job_ids if j != own]
    sched.cancel(job_ids)
    return len(job_ids)


def cmd_cancel(args):
    batch_dir, cfg = args.dir, load_config(args.dir)
    if not args.ligands and not args.all:
        sys.exit("Name the ligands to cancel, or use --all")
    sched = Scheduler(cfg)
    with BatchLock(batch_dir):
        for lig in select_ligands(batch_dir, args.ligands):
            state = read_state(batch_dir, lig)
            if state["state"] in (QUEUED, RUNNING):
                write_state(batch_dir, lig, state=CANCELLED)
            n = kill_ligand_jobs(sched, batch_dir, lig)
            if n or state["state"] in (QUEUED, RUNNING):
                print(f"  {lig}: cancelled {n} job(s)")


def cmd_hold(args):
    """Pause the batch: nothing new is started until 'release'."""
    batch_dir, cfg = args.dir, load_config(args.dir)
    sched = Scheduler(cfg)
    with BatchLock(batch_dir):
        active_jobs = sched.active_jobs()
        # Remember the ligands which are active now: 'release' restarts those
        # whose driver did not survive the pause
        active = [
            lig
            for lig in list_ligands(batch_dir)
            if effective_state(read_state(batch_dir, lig), active_jobs)
            in (QUEUED, RUNNING)
        ]
        previous = read_hold(batch_dir)
        info = {
            "since": previous.get("since") or now(),
            "ligands": sorted(set(previous.get("ligands", [])) | set(active)),
        }
        with open(hold_path(batch_dir) + ".tmp", "w") as f:
            json.dump(info, f, indent=1)
        os.replace(hold_path(batch_dir) + ".tmp", hold_path(batch_dir))
    # The drivers stop submitting once they see the marker. Hold twice, to
    # catch the jobs which were being submitted at that moment.
    held, killed = set(), set()
    for round_no in range(2):
        if round_no:
            time.sleep(args.settle)
            active_jobs = sched.active_jobs()
        sims, drivers, running = batch_jobs(batch_dir, active_jobs)
        sched.hold(sims + drivers)
        held.update(sims + drivers)
        if args.kill_running:
            sched.cancel(running)
            killed.update(running)
    print(f"The batch is on hold: {len(held)} waiting job(s) held.")
    if args.kill_running:
        print(
            f"{len(killed)} running simulation job(s) deleted; they are repeated "
            "from their start after the release."
        )
    elif running:
        print(
            f"{len(running)} simulation job(s) are still running and will finish "
            "(delete them with 'hold --kill-running' to free their nodes now)."
        )
    print(f"Continue with '{PROGRAM} release'.")


def cmd_release(args):
    """Continue a paused batch, restarting ligands whose driver died meanwhile."""
    batch_dir, cfg = args.dir, load_config(args.dir)
    sched = Scheduler(cfg)
    if not is_held(batch_dir):
        sys.exit("The batch is not on hold.")
    info = read_hold(batch_dir)
    os.remove(hold_path(batch_dir))
    sims, drivers, _ = batch_jobs(batch_dir, sched.active_jobs())
    sched.release(sims + drivers)
    print(f"Released {len(sims) + len(drivers)} job(s).")
    # Ligands which were active when the batch was paused but have no driver
    # any more (walltime, node or server failure) are resumed. Their left-over
    # simulation jobs are replaced by the new driver.
    max_active = args.max_active or cfg["max_active_ligands"]
    submitted, _, waiting, skipped = submit_ligands(
        batch_dir, cfg, sched, list_ligands(batch_dir), max_active, False,
        resume=info.get("ligands", []),
    )  # fmt: skip
    if submitted:
        print(f"Restarted {len(submitted)} ligand(s): {' '.join(submitted)}")
    if waiting > 0:
        print(
            f"{waiting} more will be started automatically as running ligands finish."
        )
    if skipped:
        print(
            f"{len(skipped)} interrupted ligand(s) were not resumed because they already "
            f"stopped {cfg['max_attempts']} times - resume with 'submit --retry'."
        )


def cmd_reset(args):
    """Delete everything a3fe produced for the named ligands, keeping the inputs."""
    batch_dir, cfg = args.dir, load_config(args.dir)
    sched = Scheduler(cfg)
    keep = ("ligand.sdf", "protein.pdb", "water.parm7", "water.rst7")
    if args.keep_parameters:
        keep += (
            "protein.parm7", "protein.rst7", "lig_param.prm7", "lig_param.rst7",
            "bound_param.prm7", "bound_param.rst7", "free_param.prm7", "free_param.rst7",
        )  # fmt: skip
    for lig in select_ligands(batch_dir, args.ligands):
        kill_ligand_jobs(sched, batch_dir, lig)
        lig_dir = os.path.join(batch_dir, lig)
        for entry in os.listdir(lig_dir):
            path = os.path.join(lig_dir, entry)
            if entry == "input":
                for f in os.listdir(path):
                    if f not in keep:
                        fp = os.path.join(path, f)
                        shutil.rmtree(fp) if os.path.isdir(fp) else os.remove(fp)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        print(f"  {lig}: reset")


# --------------------------------------------------------------------------- #
# status / collect
# --------------------------------------------------------------------------- #


def parse_time(timestamp):
    try:
        return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def duration(start, end=None):
    """The time between two timestamps (until now by default), e.g. '2h05m'."""
    start, end = parse_time(start), parse_time(end) if end else datetime.datetime.now()
    if start is None or end is None:
        return "-"
    minutes = max(int((end - start).total_seconds() // 60), 0)
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 48 * 60:
        return f"{minutes // 60}h{minutes % 60:02d}m"
    return f"{minutes // 1440}d"


def age(timestamp):
    return duration(timestamp)


def elapsed(state, effective):
    """The wall time of a ligand: from the first start of its driver until it
    finished (or until now if it is running, or until it stopped otherwise).
    This includes the time its jobs waited in the queue."""
    start = state.get("first_started") or state.get("started")
    if effective in (NEW, QUEUED) and not state.get("first_started"):
        return "-"
    if effective == RUNNING:
        return duration(start)
    end = state.get("finished") if effective == DONE else state.get("updated")
    return duration(start, end) if end else "-"


def cmd_status(args):
    batch_dir, cfg = args.dir, load_config(args.dir)
    sched = Scheduler(cfg)
    active_jobs = sched.active_jobs()
    if len(args.ligands) == 1:
        return status_detail(
            batch_dir, select_ligands(batch_dir, args.ligands)[0], active_jobs
        )

    sim_jobs = {}
    for name, state in active_jobs.values():
        counts = sim_jobs.setdefault(name, {"R": 0, "Q": 0})
        counts[state] += 1

    rows, totals = [], {}
    for lig in select_ligands(batch_dir, args.ligands):
        state = read_state(batch_dir, lig)
        eff = effective_state(state, active_jobs)
        totals[eff] = totals.get(eff, 0) + 1
        if args.state and eff not in args.state:
            continue
        stage = state.get("stage", "-") if eff not in (NEW, DONE) else "-"
        done, total = state.get("sims_done"), state.get("sims_total")
        sims = f"{done}/{total}" if total else "-"
        jobs = sim_jobs.get(sim_job_name(batch_dir, lig), {"R": 0, "Q": 0})
        result = read_result(batch_dir, lig)
        dg = f"{result[0]} +/- {result[1]}" if result else "-"
        note = (state.get("error") or "").splitlines()[0][:60] if eff == FAILED else ""
        rows.append(
            (
                lig,
                eff,
                stage,
                sims,
                f"{jobs['R']}/{jobs['Q']}",
                dg,
                elapsed(state, eff),
                age(state.get("updated")),
                note,
            )
        )

    header = (
        "LIGAND",
        "STATE",
        "STAGE",
        "SIMS",
        "JOBS R/Q",
        "dG (kcal/mol)",
        "ELAPSED",
        "UPDATED",
        "",
    )
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print(f"\n  {os.path.basename(os.path.abspath(batch_dir))}   {now()}\n")
    for row in [header] + rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip())
    n = sum(totals.values())
    summary = ", ".join(f"{count} {name}" for name, count in sorted(totals.items()))
    running = sum(s == "R" for _, s in active_jobs.values())
    print(f"\n  {n} ligands: {summary}")
    if is_held(batch_dir):
        print(
            f"  ON HOLD since {read_hold(batch_dir).get('since', '?')} - continue "
            f"with '{PROGRAM} release'"
        )
    print(
        f"  your jobs in the queue: {running} running, {len(active_jobs) - running} waiting\n"
    )


def status_detail(batch_dir, lig, active_jobs):
    state = read_state(batch_dir, lig)
    eff = effective_state(state, active_jobs)
    lig_dir = os.path.join(batch_dir, lig)
    print(f"\n  {lig}: {eff}")
    for key in (
        "stage",
        "job_id",
        "host",
        "attempts",
        "submitted",
        "started",
        "updated",
    ):
        if state.get(key) is not None:
            print(f"    {key:10s} {state[key]}")
    print(f"    {'elapsed':10s} {elapsed(state, eff)}")
    result = read_result(batch_dir, lig)
    if result:
        print(
            f"    dG         {result[0]} +/- {result[1]} kcal/mol  (runs: {', '.join(result[2])})"
        )
    if state.get("error"):
        print("\n  Error:\n    " + state["error"].strip().replace("\n", "\n    "))

    # Simulations of each stage. A simulation counts as finished once its final
    # output has been written (approximate - the driver does the strict check).
    stage_dirs = sorted(glob.glob(os.path.join(lig_dir, "*", "*", "output")))
    if stage_dirs:
        print("\n  Production simulations (finished / total):")
    for stage_dir in stage_dirs:
        runs = glob.glob(os.path.join(stage_dir, "lambda_*", "run_*"))
        finished = [
            r
            for r in runs
            if os.path.isfile(os.path.join(r, "prod", "prod.gro"))
            or os.path.isfile(os.path.join(r, "simfile.dat"))
        ]
        leg, stage = stage_dir.split(os.sep)[-3:-1]
        print(f"    {leg + '/' + stage:18s} {len(finished):4d} / {len(runs)}")

    name = sim_job_name(batch_dir, lig)
    jobs = [(j, s) for j, (n, s) in active_jobs.items() if n == name]
    print(
        f"\n  Simulation jobs in the queue: {sum(s == 'R' for _, s in jobs)} running, "
        f"{sum(s == 'Q' for _, s in jobs)} waiting"
    )
    log = os.path.join(lig_dir, STATE_DIR, "driver.log")
    if os.path.isfile(log):
        with open(log, errors="replace") as f:
            lines = [line.rstrip() for line in f.readlines() if "DEBUG" not in line]
        print(f"\n  Last lines of {log}:")
        for line in lines[-15:]:
            print("    " + line[:200])
    print()


def cmd_collect(args):
    batch_dir = args.dir
    rows = []
    for lig in list_ligands(batch_dir):
        result = read_result(batch_dir, lig)
        if result:
            rows.append([lig[len(LIG_PREFIX) :], result[0], result[1]] + result[2])
    n_runs = max((len(r) - 3 for r in rows), default=0)
    out = args.output or os.path.join(batch_dir, "abfe_results.tsv")
    with open(out, "w") as f:
        f.write(
            "\t".join(["id", "dG", "CI95"] + [f"run{i + 1}" for i in range(n_runs)])
            + "\n"
        )
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(
        f"{len(rows)} of {len(list_ligands(batch_dir))} ligands have results -> {out}"
    )


# --------------------------------------------------------------------------- #
# run: the driver of one ligand (executed inside the driver job)
# --------------------------------------------------------------------------- #


def limit_cpu_count():
    """Make multiprocessing use the cores given to this job. a3fe creates
    process pools with one worker per core of the *node*, which exhausts
    the memory of the job on large nodes."""
    n = len(os.sched_getaffinity(0))
    os.cpu_count = lambda: n
    if hasattr(os, "process_cpu_count"):
        os.process_cpu_count = lambda: n
    return n


def run_checked(cmd, cwd, **kwargs):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed in {cwd}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result


def prepare_inputs(input_dir):
    """Parameterise the protein (ff14SB) and the ligand (OpenFF 2.0.0) and
    assemble the bound and free systems, unless this was already done.

    Needs ligand.sdf and protein.pdb (or protein.parm7 + protein.rst7).
    Optional crystallographic waters: water.parm7 + water.rst7."""
    final = [
        "bound_param.prm7",
        "bound_param.rst7",
        "free_param.prm7",
        "free_param.rst7",
    ]
    if all(os.path.isfile(os.path.join(input_dir, f)) for f in final):
        print("Parameterised input found - skipping the preparation", flush=True)
        return
    path = lambda name: os.path.join(input_dir, name)  # noqa: E731

    if not (
        os.path.isfile(path("protein.parm7")) and os.path.isfile(path("protein.rst7"))
    ):
        if not os.path.isfile(path("protein.pdb")):
            raise FileNotFoundError(
                f"No protein.pdb (or protein.parm7/.rst7) in {input_dir}"
            )
        print("Parameterising the protein (pdb4amber + tleap, ff14SB)", flush=True)
        run_checked(
            ["pdb4amber", "-i", "protein.pdb", "-o", "protein_sanitised.pdb"], input_dir
        )
        leap = (
            "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
            "prot = loadpdb protein_sanitised.pdb\n"
            "saveamberparm prot protein.parm7 protein.rst7\nquit\n"
        )
        result = run_checked(["tleap", "-f", "-"], input_dir, input=leap)
        if (
            not os.path.isfile(path("protein.rst7"))
            or os.path.getsize(path("protein.rst7")) == 0
        ):
            raise RuntimeError(
                f"tleap could not parameterise the protein:\n{result.stdout[-3000:]}"
            )

    import BioSimSpace as BSS

    print("Parameterising the ligand (OpenFF 2.0.0)", flush=True)
    ligand = BSS.IO.readMolecules(path("ligand.sdf"))[0]
    ligand = BSS.Parameters.openff_unconstrained_2_0_0(ligand).getMolecule()
    BSS.IO.saveMolecules(path("lig_param"), ligand, ["prm7", "rst7"])

    print("Assembling the bound and free systems", flush=True)
    protein = BSS.IO.readMolecules([path("protein.parm7"), path("protein.rst7")])
    ligand = BSS.IO.readMolecules([path("lig_param.prm7"), path("lig_param.rst7")])
    bound = ligand + protein
    if os.path.isfile(path("water.parm7")) and os.path.isfile(path("water.rst7")):
        bound = bound + BSS.IO.readMolecules([path("water.parm7"), path("water.rst7")])
    # Write under temporary names, so that an interrupted preparation is repeated
    BSS.IO.saveMolecules(path("tmp_bound_param"), bound, ["prm7", "rst7"])
    BSS.IO.saveMolecules(path("tmp_free_param"), ligand, ["prm7", "rst7"])
    for name in final:
        tmp = path("tmp_" + name)
        if name.endswith(".prm7"):
            # a3fe expects the ligand residue to be named LIG
            with open(tmp) as f:
                content = f.read().replace("MOL ", "LIG ")
            with open(tmp, "w") as f:
                f.write(content)
    for name in final:
        os.replace(path("tmp_" + name), path(name))


def install_qsub_wrapper(lig_dir):
    """
    Put a wrapper of qsub first on the PATH, so that the jobs submitted by a3fe
    are resubmitted while the PBS server hands out job ids which are still in
    use (see PBS_ID_COLLISION) or can not be reached (see SERVER_ERRORS).
    Without it a3fe gives up after 15 attempts.
    """
    bin_dir = os.path.join(lig_dir, STATE_DIR, "bin")
    path = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d != bin_dir]
    qsub = shutil.which("qsub", path=os.pathsep.join(path))
    if qsub is None:
        return
    os.makedirs(bin_dir, exist_ok=True)
    wrapper = os.path.join(bin_dir, "qsub")
    server_errors = "|".join(SERVER_ERRORS)
    script = f"""#!/bin/bash
# Written by a3fe-abfe: repeat qsub while the job id is already in use or the
# PBS server can not be reached
ids=0; conn=0
server_errors={shlex.quote(server_errors)}
while true; do
    out=$({shlex.quote(qsub)} "$@" 2>&1); rc=$?
    [ $rc -eq 0 ] && break
    if [[ "$out" == *"{PBS_ID_COLLISION}"* ]] && [ $ids -lt {PBS_ID_RETRIES} ]; then
        ids=$((ids + 1)); sleep {PBS_ID_RETRY_WAIT}
    elif [[ "$out" =~ $server_errors ]] && [ $conn -lt {SERVER_RETRIES} ]; then
        conn=$((conn + 1)); sleep {SERVER_RETRY_WAIT}
    else
        break
    fi
done
printf '%s\\n' "$out"
exit $rc
"""
    with open(wrapper, "w") as f:
        f.write(script)
    os.chmod(wrapper, 0o755)
    os.environ["PATH"] = os.pathsep.join([bin_dir] + path)


def tolerate_lost_jobs():
    """
    Make a3fe treat a job which left the queue without writing its output file
    (e.g. it was lost by the scheduler when its node had a problem) as a
    finished job, instead of raising an error which stops the whole ligand.
    The simulation is then found to be incomplete and is repeated.
    (Only in this process, the a3fe code is not changed.)
    """
    from a3fe.run import _virtual_queue

    original = _virtual_queue.Job.has_failed
    if getattr(original, "_tolerant", False):
        return

    def has_failed(self):
        try:
            return original(self)
        except FileNotFoundError:
            print(
                f"[{now()}] Job {self.slurm_job_id} ended without any output "
                f"(expected {self.slurm_file_base}*) - it will be repeated",
                flush=True,
            )
            return False

    has_failed._tolerant = True
    _virtual_queue.Job.has_failed = has_failed


def drop_slurm_time_limit():
    """
    Make a3fe write SLURM scripts without a time limit. SlurmConfig always
    writes "#SBATCH --time", so the line is removed from the scripts it writes
    (only in this process, the a3fe code is not changed).
    """
    import a3fe as a3

    original = a3.SlurmConfig.get_submission_cmds
    if getattr(original, "_no_time_limit", False):
        return

    def get_submission_cmds(self, *args, **kwargs):
        cmds = original(self, *args, **kwargs)
        with open(cmds[-1]) as f:
            lines = f.readlines()
        with open(cmds[-1], "w") as f:
            f.writelines(
                line for line in lines if not line.startswith("#SBATCH --time=")
            )
        return cmds

    get_submission_cmds._no_time_limit = True
    a3.SlurmConfig.get_submission_cmds = get_submission_cmds


def make_scheduler_config(cfg, batch_dir, lig):
    """The a3fe scheduler configuration used for all jobs of a ligand."""
    import a3fe as a3

    res = cfg["simulation"]
    extra = dict(res["extra_options"])
    if cfg["scheduler"] == "pbs":
        if not hasattr(a3, "PbsConfig"):
            raise RuntimeError(
                "This a3fe installation has no PBS support (a3fe.PbsConfig)"
            )
        extra["N"] = sim_job_name(batch_dir, lig)
        config = a3.PbsConfig(
            walltime=res["walltime"],
            ncpus=res["ncpus"],
            ngpus=res["ngpus"],
            mem=res["mem"],
            extra_options=extra,
        )
        config.queue = res["queue"] or a3.PbsConfig.get_default_partition()
        # Modules etc. first, then the conda environment (activated by default)
        config.setup_commands = list(cfg["setup_commands"]) + config.setup_commands
        return config

    extra["job-name"] = sim_job_name(batch_dir, lig)
    extra.setdefault("cpus-per-task", str(res["ncpus"]))
    if res["mem"]:
        extra.setdefault("mem", res["mem"])
    kwargs = dict(
        gres=f"gpu:{res['ngpus']}" if res["ngpus"] else "none",
        extra_options=extra,
    )
    # Some a3fe versions accept commands to set up the environment of SLURM jobs
    if "setup_commands" in a3.SlurmConfig.model_fields:
        kwargs["setup_commands"] = list(cfg["setup_commands"])
    kwargs["partition"] = res["queue"] or a3.SlurmConfig.get_default_partition()
    if res["walltime"]:
        kwargs["time"] = res["walltime"]
    else:
        drop_slurm_time_limit()
    return a3.SlurmConfig(**kwargs)


def make_sysprep_config(cfg, engine_type):
    import a3fe as a3

    protocol = cfg["protocol"]
    sysprep = engine_type.system_prep_config()
    sysprep.slurm = True
    for key, value in protocol["sysprep"].items():
        setattr(sysprep, key, value)
    if protocol["lambda_values"]:
        sysprep.lambda_values = {
            a3.LegType[leg.upper()]: {
                a3.StageType[stage.upper()]: [float(v) for v in values]
                for stage, values in stages.items()
            }
            for leg, stages in protocol["lambda_values"].items()
        }
    return sysprep


def all_stages(calc):
    return [stage for leg in calc.legs for stage in leg.stages]


def all_sims(calc):
    return [
        sim
        for stage in all_stages(calc)
        for win in stage.lam_windows
        for sim in win.sims
    ]


def setup_calculation(calc, sysprep, lig_dir):
    """Like calc.setup(), but resumable: a leg whose setup has completed is
    reloaded, and a leg whose setup was interrupted is set up again from the
    (kept) solvated / minimised / pre-equilibrated input files."""
    from a3fe import Leg

    if calc.setup_complete:
        print("Setup already complete", flush=True)
        return
    calc._validate_input(sysprep)
    calc.legs = []
    for leg_type in reversed(sysprep.required_legs):
        leg_dir = os.path.join(lig_dir, leg_type.name.lower())
        marker = os.path.join(leg_dir, ".setup_done")
        if not os.path.isfile(marker) and os.path.isdir(leg_dir):
            print(
                f"Removing the incomplete setup of the {leg_type.name.lower()} leg",
                flush=True,
            )
            shutil.rmtree(leg_dir)
        leg = Leg(
            leg_type=leg_type,
            equil_detection=calc.equil_detection,
            runtime_constant=calc.runtime_constant,
            relative_simulation_cost=calc.relative_simulation_cost,
            ensemble_size=calc.ensemble_size,
            input_dir=calc.input_dir,
            base_dir=leg_dir,
            stream_log_level=calc.stream_log_level,
            slurm_config=calc.slurm_config,
            analysis_slurm_config=calc.analysis_slurm_config,
            engine_config=calc.engine_config,
            engine_type=calc.engine_type,
        )
        calc.legs.append(leg)
        if not os.path.isfile(marker):
            print(f"[{now()}] Setting up the {leg_type.name.lower()} leg", flush=True)
            leg.setup(sysprep)
            open(marker, "w").close()
    calc.setup_complete = True
    calc._dump()


def run_production(calc, cfg, progress, batch_dir=None):
    """Run all production simulations which have not completed yet, and
    wait for them. Simulations which fail are repeated a few times.

    While the batch is on hold nothing is submitted, and simulations which
    were deleted during the hold are repeated without counting as failures."""

    def held():
        return batch_dir is not None and is_held(batch_dir)

    protocol = cfg["protocol"]
    runtime = float(protocol["runtime"])
    sims = all_sims(calc)
    queues = []
    for stage in all_stages(calc):
        if stage.virtual_queue not in queues:
            queues.append(stage.virtual_queue)

    def complete(sim):
        try:
            return (
                sim.engine_backend.run_completed(sim.output_dir, sim.slurm_output_files)
                and sim.get_tot_simtime() >= runtime * 0.999
            )
        except Exception:
            return False

    interval = calc.slurm_config.queue_check_interval
    attempt = 0
    while attempt <= protocol["max_sim_retries"]:
        if batch_dir is not None:
            wait_while_held(batch_dir)
        todo = [sim for sim in sims if not complete(sim)]
        progress(len(sims) - len(todo), len(sims))
        if not todo:
            return
        print(
            f"[{now()}] Production attempt {attempt + 1}: running {len(todo)} of "
            f"{len(sims)} simulations for {runtime} ns",
            flush=True,
        )
        was_held = False
        waiting = []
        for sim in todo:
            if held():
                # The remaining simulations are submitted after the release
                was_held = True
                break
            # Remove any partial output, so that the simulation starts afresh
            sim.clean()
            sim.run(runtime)
            waiting.append(sim)
            # Do not flood the scheduler with submissions
            time.sleep(float(cfg["submit_delay"]))
        n_finished = len(sims) - len(todo) + len(waiting)
        while waiting:
            time.sleep(interval)
            was_held = was_held or held()
            for queue in queues:
                queue.update()
            # Checking sim.running also resubmits simulations which crashed with NaNs
            waiting = [sim for sim in waiting if sim.running]
            progress(n_finished - len(waiting), len(sims))
        # Simulations lost in a round with a hold were probably deleted on purpose
        if not was_held:
            attempt += 1

    failed = [sim.output_dir for sim in sims if not complete(sim)]
    if failed:
        listing = "\n".join(failed[:10]) + ("\n..." if len(failed) > 10 else "")
        raise RuntimeError(
            f"{len(failed)} simulation(s) did not complete after "
            f"{protocol['max_sim_retries'] + 1} attempts. Check the output in:\n{listing}"
        )


def cmd_run(args):
    batch_dir, cfg = args.dir, load_config(args.dir)
    lig = select_ligands(batch_dir, [args.ligand])[0]
    lig_dir = os.path.join(batch_dir, lig)
    sched = Scheduler(cfg)
    job_id = (
        os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or ""
    ).split(".")[0]

    def update(**changes):
        write_state(batch_dir, lig, **changes)

    def stop(signum, frame):
        raise Interrupted(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    # first_started is kept when a ligand is resumed, to report its total wall time
    first_started = read_state(batch_dir, lig).get("first_started") or now()
    update(state=RUNNING, stage="prepare", job_id=job_id or None, host=os.uname().nodename,
           started=now(), first_started=first_started, finished=None, error=None)  # fmt: skip
    print(
        f"\n[{now()}] ===== {lig}: driver started (job {job_id or 'none'}) =====",
        flush=True,
    )
    try:
        ncpus = limit_cpu_count()
        print(f"Using {ncpus} core(s) for the driver", flush=True)
        wait_while_held(batch_dir)
        # Remove simulation jobs left behind by an interrupted driver
        n = kill_ligand_jobs(sched, batch_dir, lig, driver=False)
        if n:
            print(f"Cancelled {n} job(s) left over from a previous run", flush=True)
            time.sleep(10)

        prepare_inputs(os.path.join(lig_dir, "input"))
        if cfg["scheduler"] == "pbs":
            install_qsub_wrapper(lig_dir)

        import logging

        import a3fe as a3

        update(stage="setup")
        protocol = cfg["protocol"]
        engine_type = a3.EngineType[cfg["engine"].upper()]
        sched_config = make_scheduler_config(cfg, batch_dir, lig)
        engine_config = None
        if protocol["engine_options"]:
            engine_class = (
                a3.GromacsConfig if cfg["engine"] == "gromacs" else a3.SomdConfig
            )
            engine_config = engine_class(**protocol["engine_options"])
        os.chdir(lig_dir)
        calc = a3.Calculation(
            base_dir=lig_dir,
            input_dir=os.path.join(lig_dir, "input"),
            ensemble_size=protocol["ensemble_size"],
            slurm_config=sched_config,
            engine_config=engine_config,
            engine_type=engine_type,
            stream_log_level=logging.INFO,
        )

        def apply_scheduler_config():
            """Apply the current scheduler settings, which may have changed
            since the calculation was saved."""
            calc.recursively_set_attr(
                "slurm_config", sched_config, force=True, silent=True
            )
            calc.recursively_set_attr(
                "analysis_slurm_config",
                sched_config.model_copy(),
                force=True,
                silent=True,
            )

        apply_scheduler_config()
        tolerate_lost_jobs()
        sysprep = make_sysprep_config(cfg, engine_type)
        attempt = 0
        while True:
            try:
                setup_calculation(calc, sysprep, lig_dir)
                break
            except Interrupted:
                raise
            except Exception as e:
                # A set-up step whose jobs were deleted during a hold is
                # repeated after the release, without counting as a failure
                if not wait_while_held(batch_dir):
                    attempt += 1
                if attempt > protocol["max_sim_retries"]:
                    raise
                traceback.print_exc()
                print(
                    f"[{now()}] The setup failed ({type(e).__name__}) - repeating the "
                    f"unfinished part (attempt {attempt + 1})",
                    flush=True,
                )
                kill_ligand_jobs(sched, batch_dir, lig, driver=False)
                time.sleep(30)
        apply_scheduler_config()
        # Forget the jobs of previous, interrupted runs
        for runner in calc.legs + all_stages(calc):
            runner.virtual_queue._flush()

        update(stage="production")
        run_production(
            calc,
            cfg,
            lambda done, total: update(sims_done=done, sims_total=total),
            batch_dir,
        )

        update(stage="analysis")
        print(f"[{now()}] Analysing", flush=True)
        calc.set_equilibration_time(float(protocol["equilibration_time"]))
        calc.analyse(slurm=False, plot_rmsds=False)
        if protocol["lighten"]:
            calc.lighten()
        calc.save()
        result = read_result(batch_dir, lig)
        if result is None:
            raise RuntimeError("The analysis did not write output/overall_stats.dat")
        update(state=DONE, finished=now())
        print(
            f"[{now()}] ===== {lig}: done, dG = {result[0]} +/- {result[1]} kcal/mol =====",
            flush=True,
        )
    except Interrupted as e:
        # Leave a state from which 'submit' resumes - unless the ligand was cancelled
        if read_state(batch_dir, lig).get("state") != CANCELLED:
            update(state=INTERRUPTED, error=f"Interrupted: {e}")
        print(f"[{now()}] ===== {lig}: interrupted ({e}) =====", flush=True)
        kill_ligand_jobs(sched, batch_dir, lig, driver=False)
        sys.exit(1)
    except BaseException as e:
        traceback.print_exc()
        update(state=FAILED, error=f"{type(e).__name__}: {e}")
        print(f"[{now()}] ===== {lig}: FAILED =====", flush=True)
        kill_ligand_jobs(sched, batch_dir, lig, driver=False)
        sys.exit(1)


# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d", "--dir", default=".", help="batch directory (default: current)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "init", help="create ligand directories from an SDF file and a protein"
    )
    p.add_argument("--protein", required=True, help="prepared protein (PDB)")
    p.add_argument(
        "--ligands", required=True, help="SDF file with the docked, protonated ligands"
    )
    p.add_argument("--water", nargs=2, metavar=("PARM7", "RST7"),
                   help="optional crystallographic waters (water.parm7 water.rst7)")  # fmt: skip
    p.set_defaults(func=cmd_init)

    p = sub.add_parser(
        "submit", help="start or resume ligands (all which are not done by default)"
    )
    p.add_argument(
        "ligands", nargs="*", help="ligands to (re)start, also if failed or cancelled"
    )
    p.add_argument(
        "--retry", action="store_true", help="also restart failed and cancelled ligands"
    )
    p.add_argument("--max-active", type=int, help="override max_active_ligands")
    p.add_argument("--from-driver", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="show the progress (give one ligand for details)")
    p.add_argument("ligands", nargs="*")
    p.add_argument("--state", nargs="+", help="only list ligands in these states",
                   choices=[NEW, QUEUED, RUNNING, DONE, FAILED, INTERRUPTED, CANCELLED])  # fmt: skip
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("cancel", help="stop ligands and delete their jobs")
    p.add_argument("ligands", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser(
        "hold", help="pause the batch: hold its waiting jobs, start nothing new"
    )
    p.add_argument("--kill-running", action="store_true",
                   help="also delete the running simulation jobs (repeated after the release)")  # fmt: skip
    p.add_argument("--settle", type=float, default=15, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_hold)

    p = sub.add_parser("release", help="continue a batch paused with 'hold'")
    p.add_argument("--max-active", type=int, help="override max_active_ligands")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser(
        "reset", help="delete all output of the named ligands (keeps the input)"
    )
    p.add_argument("ligands", nargs="+")
    p.add_argument("--keep-parameters", action="store_true",
                   help="keep the parameterised protein and ligand")  # fmt: skip
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("collect", help="write a table of the results")
    p.add_argument("-o", "--output", help="default: <batch dir>/abfe_results.tsv")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser(
        "run", help="run one ligand in the foreground (used by the driver job)"
    )
    p.add_argument("ligand")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.dir = os.path.abspath(args.dir)
    args.func(args)


if __name__ == "__main__":
    main()
