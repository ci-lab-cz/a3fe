"""Unit tests for the a3fe-abfe command line tool. No scheduler is required."""

import datetime
import json
import os
import stat
import subprocess
from unittest.mock import patch

import pytest
import yaml

from a3fe import PbsConfig, SlurmConfig
from a3fe.cli import abfe

SDF = """lig A
  test

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
$$$$
B/2
  test

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
$$$$
"""

QSTAT = """Job Id: 11.pbs-server
    Job_Name = drv.lig_A
    Job_Owner = me@pbs-login
    job_state = R
Job Id: 12.pbs-server
    Job_Name = sim.lig_A
    Job_Owner = me@pbs-login
    job_state = Q
Job Id: 13.pbs-server
    Job_Name = sim.lig_A
    Job_Owner = me@pbs-login
    job_state = F
Job Id: 14.pbs-server
    Job_Name = sim.lig_A
    Job_Owner = other@pbs-login
    job_state = R
"""


@pytest.fixture
def batch(tmp_path):
    """A batch directory with two ligands, created with 'init'."""
    sdf, protein = tmp_path / "ligands.sdf", tmp_path / "protein.pdb"
    sdf.write_text(SDF)
    protein.write_text("ATOM\n")
    batch_dir = str(tmp_path / "batch")
    abfe.main(
        ["-d", batch_dir, "init", "--protein", str(protein), "--ligands", str(sdf)]
    )
    return batch_dir


def set_state(batch_dir, lig, **state):
    abfe.write_state(batch_dir, lig, **state)


def submit(batch_dir, *args, active_jobs=None):
    """Run 'submit' with a mocked scheduler. Returns the submitted ligands."""
    submitted = []

    def submit_driver(self, batch_dir, lig):
        submitted.append(lig)
        return str(100 + len(submitted))

    with (
        patch.object(abfe.Scheduler, "active_jobs", return_value=active_jobs or {}),
        patch.object(abfe.Scheduler, "submit_driver", submit_driver),
    ):
        abfe.main(["-d", batch_dir, "submit", *args])
    return submitted


def test_init(batch):
    """Test that the ligand directories and the settings are created."""
    assert abfe.list_ligands(batch) == ["lig_B_2", "lig_lig_A"]
    assert sorted(os.listdir(os.path.join(batch, "lig_B_2", "input"))) == [
        "ligand.sdf",
        "protein.pdb",
    ]
    with open(os.path.join(batch, "lig_B_2", "input", "ligand.sdf")) as f:
        ligand = f.read()
    assert ligand.startswith("B/2\n") and ligand.count("$$$$") == 1
    # The settings template is copied and valid
    cfg = abfe.load_config(batch)
    assert (cfg["scheduler"], cfg["engine"]) == ("pbs", "gromacs")


def test_template_options():
    """Test that the settings template sets every option, and no unknown ones."""
    with open(abfe.CONFIG_TEMPLATE) as f:
        template = yaml.safe_load(f)
    abfe.merge(abfe.DEFAULTS, template)  # Exits if there are unknown options
    assert set(template) == set(abfe.DEFAULTS)
    for section in ("driver", "simulation", "protocol"):
        assert set(template[section]) == set(abfe.DEFAULTS[section])


def test_load_config_errors(batch):
    """Test that wrong settings are rejected."""
    path = os.path.join(batch, abfe.CONFIG_NAME)
    for settings in ["unknown_option: 1", "scheduler: lsf", "engine: somd"]:
        with open(path, "w") as f:
            f.write(settings + "\n")
        with pytest.raises(SystemExit):
            abfe.load_config(batch)


def test_merge():
    """Test that settings are merged with the defaults."""
    cfg = abfe.merge(
        abfe.DEFAULTS,
        {"simulation": {"ncpus": 4}, "protocol": {"sysprep": {"runtime_npt": 1}}},
    )
    assert cfg["simulation"]["ncpus"] == 4
    assert cfg["simulation"]["mem"] == abfe.DEFAULTS["simulation"]["mem"]
    assert cfg["protocol"]["sysprep"] == {"runtime_npt": 1}
    assert abfe.DEFAULTS["simulation"]["ncpus"] == 8


def test_select_ligands(batch):
    """Test that ligands can be named with and without the prefix."""
    assert abfe.select_ligands(batch, []) == ["lig_B_2", "lig_lig_A"]
    assert abfe.select_ligands(batch, ["B_2", "lig_B_2/"]) == ["lig_B_2", "lig_B_2"]
    assert abfe.select_ligands(batch, ["lig_A"]) == ["lig_lig_A"]
    with pytest.raises(SystemExit):
        abfe.select_ligands(batch, ["missing"])


def test_effective_state():
    """Test that a ligand without a driver job counts as interrupted."""
    jobs = {"5": ("drv.lig_A", "R")}
    assert abfe.effective_state({}, jobs) == abfe.NEW
    assert abfe.effective_state({"state": "running", "job_id": "5"}, jobs) == "running"
    assert abfe.effective_state({"state": "running", "job_id": 5}, jobs) == "running"
    assert (
        abfe.effective_state({"state": "queued", "job_id": "6"}, jobs) == "interrupted"
    )
    assert abfe.effective_state({"state": "failed", "job_id": "6"}, jobs) == "failed"


def test_submit_respects_limit_and_states(batch):
    """Test which ligands are started by 'submit'."""
    # Only max_active ligands are started
    assert submit(batch, "--max-active", "1") == ["lig_B_2"]
    assert abfe.read_state(batch, "lig_B_2")["state"] == abfe.QUEUED
    # The queued ligand is not started twice
    jobs = {"101": ("drv.lig_B_2", "Q")}
    assert submit(batch, active_jobs=jobs) == ["lig_lig_A"]
    jobs["102"] = ("drv.lig_lig_A", "R")
    assert submit(batch, active_jobs=jobs) == []


def test_submit_failed_and_interrupted(batch):
    """Test that failed ligands need --retry, and interrupted ones are resumed."""
    set_state(batch, "lig_B_2", state=abfe.FAILED)
    set_state(batch, "lig_lig_A", state=abfe.DONE)
    assert submit(batch) == []
    assert submit(batch, "--retry") == ["lig_B_2"]
    set_state(batch, "lig_B_2", state=abfe.CANCELLED)
    assert submit(batch, "B_2") == ["lig_B_2"]

    # The driver job of a running ligand has disappeared
    set_state(batch, "lig_B_2", state=abfe.RUNNING, job_id="7", attempts=1)
    assert submit(batch) == ["lig_B_2"]
    assert abfe.read_state(batch, "lig_B_2")["attempts"] == 2
    # ... but it is not resumed forever
    set_state(batch, "lig_B_2", state=abfe.RUNNING, job_id="7", attempts=3)
    assert submit(batch) == []


def test_active_jobs_pbs():
    """Test that the PBS queue is parsed correctly."""
    sched = abfe.Scheduler({"scheduler": "pbs"})
    with (
        patch.object(abfe.Scheduler, "_run", return_value=QSTAT),
        patch("getpass.getuser", return_value="me"),
    ):
        assert sched.active_jobs() == {
            "11": ("drv.lig_A", "R"),
            "12": ("sim.lig_A", "Q"),
        }


def test_active_jobs_slurm():
    """Test that the SLURM queue is parsed correctly."""
    sched = abfe.Scheduler({"scheduler": "slurm"})
    with patch.object(
        abfe.Scheduler, "_run", return_value="11|drv.lig_A|R\n12|sim.lig_A|PD\n"
    ):
        assert sched.active_jobs() == {
            "11": ("drv.lig_A", "R"),
            "12": ("sim.lig_A", "Q"),
        }


@pytest.mark.parametrize(
    "scheduler, output, expected, absent",
    [
        (
            "pbs",
            "21.pbs-server\n",
            ["#PBS -l select=1:ncpus=2:mem=16gb", "#PBS -A acc"],
            "walltime",
        ),
        (
            "slurm",
            "21;cluster\n",
            ["#SBATCH --mem=16gb", "#SBATCH --A=acc"],
            "--time",
        ),
    ],
)
def test_submit_driver(batch, scheduler, output, expected, absent):
    """Test the driver job script, here without a time limit."""
    cfg = abfe.merge(
        abfe.DEFAULTS,
        {
            "scheduler": scheduler,
            "setup_commands": ["module load gromacs"],
            "driver": {"walltime": None, "extra_options": {"A": "acc"}},
        },
    )
    with patch.object(abfe.Scheduler, "_run", return_value=output) as mocked_run:
        assert abfe.Scheduler(cfg).submit_driver(batch, "lig_B_2") == "21"
    script_path = os.path.join(batch, "lig_B_2", ".abfe", "driver.sh")
    assert mocked_run.call_args[0][0][-1] == script_path
    with open(script_path) as f:
        script = f.read()
    for line in expected + ["module load gromacs"]:
        assert line in script.splitlines()
    assert absent not in script
    name = abfe.driver_job_name(batch, "lig_B_2")
    assert f"-N {name}\n" in script or f"--job-name={name}\n" in script
    # The driver runs this module for the ligand, and then starts the next ones
    assert f"-m a3fe.cli.abfe -d {batch} run lig_B_2\n" in script
    assert f"-m a3fe.cli.abfe -d {batch} submit --from-driver\n" in script


def test_scheduler_run_retries():
    """Test that commands are repeated while the server can not be reached."""
    failure = subprocess.CompletedProcess(
        [], 1, "", "qstat: cannot connect to server x"
    )
    success = subprocess.CompletedProcess([], 0, "ok", "")
    other = subprocess.CompletedProcess([], 1, "", "qsub: Unknown queue")
    with patch.object(abfe, "SERVER_RETRY_WAIT", 0):
        with patch("subprocess.run", side_effect=[failure, failure, success]):
            assert abfe.Scheduler._run(["qstat"]) == "ok"
        with patch("subprocess.run", return_value=other) as mocked_run:
            with pytest.raises(RuntimeError, match="Unknown queue"):
                abfe.Scheduler._run(["qsub", "x"])
            assert mocked_run.call_count == 1


def test_qsub_wrapper(tmp_path, monkeypatch):
    """Test that the qsub wrapper repeats submissions which hit a used job id
    or an unreachable server, and passes on other errors."""
    fake_dir, count = tmp_path / "bin", tmp_path / "count"
    fake_dir.mkdir()
    fake = fake_dir / "qsub"
    fake.write_text(
        "#!/bin/bash\n"
        f"n=$(cat {count} 2>/dev/null || echo 0); echo $((n + 1)) > {count}\n"
        'if [ "$1" = bad ]; then echo "qsub: Unknown queue" >&2; exit 170; fi\n'
        'if [ $n -lt 2 ]; then echo "qsub: Job with requested ID already exists" >&2; exit 161; fi\n'
        'if [ $n -lt 3 ]; then echo "qsub: cannot connect to server x" >&2; exit 1; fi\n'
        'echo "$((9000 + n)).pbs-server"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(abfe, "PBS_ID_RETRY_WAIT", 0)
    monkeypatch.setattr(abfe, "SERVER_RETRY_WAIT", 0)

    lig_dir = str(tmp_path / "lig_A")
    abfe.install_qsub_wrapper(lig_dir)
    abfe.install_qsub_wrapper(lig_dir)  # Installing it twice must not nest it
    wrapper = os.path.join(lig_dir, ".abfe", "bin", "qsub")
    assert os.environ["PATH"].split(os.pathsep)[:2] == [
        os.path.dirname(wrapper),
        str(fake_dir),
    ]

    result = subprocess.run(["qsub", "job.sh"], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (0, "9003.pbs-server\n")
    result = subprocess.run(["qsub", "bad"], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (170, "qsub: Unknown queue\n")


def test_tolerate_lost_jobs(tmp_path, monkeypatch):
    """Test that a job without an output file is not an error."""
    from a3fe.run._virtual_queue import Job

    monkeypatch.setattr(Job, "has_failed", Job.has_failed)  # Restored after the test
    job = Job(virtual_job_id=1, command_list=["qsub", "a3fe.sh"], slurm_job_id=5)
    job.slurm_file_base = str(tmp_path / "pbs-")
    with pytest.raises(FileNotFoundError):
        job.has_failed()
    abfe.tolerate_lost_jobs()
    abfe.tolerate_lost_jobs()
    assert not job.has_failed()
    (tmp_path / "pbs-5.out").write_text("Particle coordinate is NaN\n")
    assert job.has_failed()


def test_drop_slurm_time_limit(tmp_path, monkeypatch):
    """Test that SLURM scripts can be written without a time limit."""
    monkeypatch.setattr(
        SlurmConfig, "get_submission_cmds", SlurmConfig.get_submission_cmds
    )
    config = SlurmConfig(partition="test")
    config.get_submission_cmds("echo test", str(tmp_path), "with")
    assert "#SBATCH --time=" in (tmp_path / "with.sh").read_text()
    abfe.drop_slurm_time_limit()
    cmds = config.get_submission_cmds("echo test", str(tmp_path), "without")
    script = (tmp_path / "without.sh").read_text()
    assert cmds[0] == "sbatch"
    assert "#SBATCH --time=" not in script
    assert "#SBATCH --partition=test\n" in script and "echo test" in script


def test_make_scheduler_config_pbs():
    """Test the configuration of the simulation jobs on PBS."""
    cfg = abfe.merge(
        abfe.DEFAULTS,
        {
            "setup_commands": ["module load gromacs"],
            "simulation": {
                "queue": "long",
                "walltime": None,
                "extra_options": {"A": "acc"},
            },
        },
    )
    config = abfe.make_scheduler_config(cfg, "/batch", "lig_A")
    assert isinstance(config, PbsConfig)
    assert (config.queue, config.walltime) == ("long", None)
    assert config.get_select() == "1:ncpus=8:mem=8gb"
    assert config.extra_options == {
        "A": "acc",
        "N": abfe.sim_job_name("/batch", "lig_A"),
    }
    assert config.setup_commands[0] == "module load gromacs"


def test_make_scheduler_config_slurm(monkeypatch):
    """Test the configuration of the simulation jobs on SLURM."""
    monkeypatch.setattr(
        SlurmConfig, "get_submission_cmds", SlurmConfig.get_submission_cmds
    )
    cfg = abfe.merge(
        abfe.DEFAULTS,
        {"scheduler": "slurm", "simulation": {"queue": "cpu", "walltime": "01:00:00"}},
    )
    config = abfe.make_scheduler_config(cfg, "/batch", "lig_A")
    assert isinstance(config, SlurmConfig)
    assert (config.partition, config.time, config.gres) == ("cpu", "01:00:00", "none")
    assert config.extra_options == {
        "job-name": abfe.sim_job_name("/batch", "lig_A"),
        "cpus-per-task": "8",
        "mem": "8gb",
    }


def test_status_and_collect(batch, capsys):
    """Test the status table and the table of results."""
    output_dir = os.path.join(batch, "lig_B_2", "output")
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "overall_stats.dat"), "w") as f:
        f.write(
            "Calculation over 2 runs\n"
            "Mean free energy: -4.5 + /- 1.2 kcal/mol\n"
            "Free energy from run 1: -4.0 +/- 0.1 kcal/mol\n"
            "Free energy from run 2: -5.0 +/- 0.1 kcal/mol\n"
        )
    set_state(
        batch,
        "lig_B_2",
        state=abfe.DONE,
        sims_done=4,
        sims_total=4,
        first_started="2026-01-01 10:00:00",
        started="2026-01-01 11:00:00",
        finished="2026-01-01 12:05:30",
    )
    set_state(batch, "lig_lig_A", state=abfe.RUNNING, job_id="11", stage="production")
    jobs = {
        "11": (abfe.driver_job_name(batch, "lig_lig_A"), "R"),
        "12": (abfe.sim_job_name(batch, "lig_lig_A"), "Q"),
        # A ligand of the same name in another batch
        "13": (abfe.sim_job_name(batch + "_other", "lig_lig_A"), "R"),
    }
    with patch.object(abfe.Scheduler, "active_jobs", return_value=jobs):
        abfe.main(["-d", batch, "status"])
    lines = capsys.readouterr().out.splitlines()
    assert any(
        line.split()
        == ["lig_B_2", "done", "-", "4/4", "0/0", "-4.5", "+/-", "1.2", "2h05m", "0m"]
        for line in lines
    )
    assert any(
        line.split()[:5] == ["lig_lig_A", "running", "production", "-", "0/1"]
        for line in lines
    )
    assert "  2 ligands: 1 done, 1 running" in lines

    abfe.main(["-d", batch, "collect"])
    with open(os.path.join(batch, "abfe_results.tsv")) as f:
        assert f.read() == "id\tdG\tCI95\trun1\trun2\nB_2\t-4.5\t1.2\t-4.0\t-5.0\n"


def test_state_file(batch):
    """Test that the state of a ligand is stored and updated."""
    assert abfe.read_state(batch, "lig_B_2") == {"state": abfe.NEW}
    set_state(batch, "lig_B_2", state=abfe.RUNNING, stage="setup")
    set_state(batch, "lig_B_2", stage="production")
    with open(abfe.state_path(batch, "lig_B_2")) as f:
        state = json.load(f)
    assert (state["state"], state["stage"]) == (abfe.RUNNING, "production")
    assert "updated" in state


def test_elapsed():
    """Test the wall time reported for a ligand."""
    start = (datetime.datetime.now() - datetime.timedelta(minutes=90)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    assert abfe.elapsed({}, abfe.NEW) == "-"
    assert abfe.elapsed({"first_started": start}, abfe.RUNNING) == "1h30m"
    # Without first_started (older state files) the last start is used
    assert abfe.elapsed({"started": start}, abfe.RUNNING) == "1h30m"
    stopped = {
        "first_started": "2026-01-01 10:00:00",
        "updated": "2026-01-01 10:20:00",
        "finished": None,
    }
    assert abfe.elapsed(stopped, abfe.FAILED) == "20m"
    assert abfe.elapsed(stopped, abfe.INTERRUPTED) == "20m"
    # A resumed ligand which waits in the queue again keeps its time so far
    assert abfe.elapsed(stopped, abfe.QUEUED) == "20m"
    done = dict(stopped, finished="2026-01-03 10:00:00")
    assert abfe.elapsed(done, abfe.DONE) == "2d"


def test_job_names_differ_between_batches(batch):
    """Test that ligands of the same name in two batches have different jobs."""
    other = batch + "_other"
    assert abfe.sim_job_name(batch, "lig_A") != abfe.sim_job_name(other, "lig_A")
    assert abfe.driver_job_name(batch, "lig_A") != abfe.driver_job_name(other, "lig_A")
    assert abfe.sim_job_name(batch, "lig_A").startswith("sim.lig_A.")
    # The name does not depend on how the directory is given
    assert abfe.sim_job_name(batch + "/", "lig_A") == abfe.sim_job_name(batch, "lig_A")

    jobs = {
        "1": (abfe.sim_job_name(batch, "lig_B_2"), "Q"),
        "2": (abfe.driver_job_name(batch, "lig_B_2"), "R"),
        "3": (abfe.sim_job_name(other, "lig_B_2"), "Q"),
        "4": (abfe.driver_job_name(other, "lig_B_2"), "R"),
    }
    sched = abfe.Scheduler({"scheduler": "pbs"})
    with (
        patch.object(abfe.Scheduler, "active_jobs", return_value=jobs),
        patch.object(abfe.Scheduler, "cancel") as cancel,
    ):
        assert abfe.kill_ligand_jobs(sched, batch, "lig_B_2") == 2
    assert sorted(cancel.call_args[0][0]) == ["1", "2"]


def hold_release(batch_dir, command, *args, active_jobs=None):
    """Run 'hold' or 'release' with a mocked scheduler. Returns the ids of the
    held, released and cancelled jobs, and the ligands whose driver was submitted."""
    calls = {"hold": [], "release": [], "cancel": [], "submitted": []}

    def submit_driver(self, batch_dir, lig):
        calls["submitted"].append(lig)
        return str(200 + len(calls["submitted"]))

    with (
        patch.object(abfe.Scheduler, "active_jobs", return_value=active_jobs or {}),
        patch.object(abfe.Scheduler, "submit_driver", submit_driver),
        patch.object(abfe.Scheduler, "hold", lambda self, ids: calls["hold"].extend(ids)),
        patch.object(abfe.Scheduler, "release", lambda self, ids: calls["release"].extend(ids)),
        patch.object(abfe.Scheduler, "cancel", lambda self, ids: calls["cancel"].extend(ids)),
    ):  # fmt: skip
        extra = ["--settle", "0"] if command == "hold" else []
        abfe.main(["-d", batch_dir, command, *extra, *args])
    return calls


def test_hold_and_release(batch, capsys):
    """Test that 'hold' pauses the batch and 'release' continues it."""
    set_state(batch, "lig_B_2", state=abfe.RUNNING, job_id="1")
    set_state(batch, "lig_lig_A", state=abfe.QUEUED, job_id="2")
    jobs = {
        "1": (abfe.driver_job_name(batch, "lig_B_2"), "R"),
        "2": (abfe.driver_job_name(batch, "lig_lig_A"), "Q"),
        "3": (abfe.sim_job_name(batch, "lig_B_2"), "Q"),
        "4": (abfe.sim_job_name(batch, "lig_B_2"), "R"),
        "5": (abfe.sim_job_name(batch + "_other", "lig_B_2"), "Q"),
    }
    calls = hold_release(batch, "hold", active_jobs=jobs)
    assert abfe.is_held(batch)
    # Waiting jobs of this batch are held; running jobs are left alone
    assert sorted(set(calls["hold"])) == ["2", "3"]
    assert calls["cancel"] == []
    assert abfe.read_hold(batch)["ligands"] == ["lig_B_2", "lig_lig_A"]

    calls = hold_release(batch, "hold", "--kill-running", active_jobs=jobs)
    assert sorted(set(calls["cancel"])) == ["4"]

    # Nothing is submitted while the batch is on hold
    with pytest.raises(SystemExit):
        submit(batch, "--retry", active_jobs=jobs)
    assert submit(batch, "--from-driver", active_jobs=jobs) == []
    with patch.object(abfe.Scheduler, "active_jobs", return_value=jobs):
        abfe.main(["-d", batch, "status"])
    assert "ON HOLD" in capsys.readouterr().out

    calls = hold_release(batch, "release", active_jobs=jobs)
    assert not abfe.is_held(batch)
    assert sorted(calls["release"]) == ["2", "3"]
    assert calls["submitted"] == []
    with pytest.raises(SystemExit):
        hold_release(batch, "release", active_jobs=jobs)


def test_release_restarts_dead_drivers(batch):
    """Test that ligands whose driver died during the hold are resumed."""
    set_state(batch, "lig_B_2", state=abfe.RUNNING, job_id="1", attempts=3)
    set_state(batch, "lig_lig_A", state=abfe.RUNNING, job_id="2")
    jobs = {
        "1": (abfe.driver_job_name(batch, "lig_B_2"), "R"),
        "2": (abfe.driver_job_name(batch, "lig_lig_A"), "R"),
        "3": (abfe.sim_job_name(batch, "lig_B_2"), "Q"),
    }
    hold_release(batch, "hold", active_jobs=jobs)
    # One driver failed (e.g. the server could not be reached), the other one
    # disappeared, although it was already resumed max_attempts times
    set_state(batch, "lig_lig_A", state=abfe.FAILED, error="ValueError")
    calls = hold_release(batch, "release", active_jobs={"3": jobs["3"]})
    assert calls["release"] == ["3"]
    assert calls["submitted"] == ["lig_B_2", "lig_lig_A"]
    assert abfe.read_state(batch, "lig_lig_A")["state"] == abfe.QUEUED

    # A ligand which was cancelled during the hold stays cancelled
    hold_release(batch, "hold", active_jobs={"201": jobs["1"], "202": jobs["2"]})
    set_state(batch, "lig_B_2", state=abfe.CANCELLED)
    calls = hold_release(batch, "release", active_jobs={"202": jobs["2"]})
    assert calls["submitted"] == []


class FakeSim:
    """A simulation which completes when its job leaves the queue, unless the
    job was deleted."""

    def __init__(self, log):
        self.log, self.done, self.job_alive, self.deleted = log, False, False, False
        self.engine_backend = self
        self.output_dir, self.slurm_output_files = "out", []

    def run_completed(self, output_dir, files):
        return self.done

    def get_tot_simtime(self):
        return 1.0

    def clean(self):
        pass

    def run(self, runtime):
        self.log.append("run")
        self.job_alive = True

    @property
    def running(self):
        if self.job_alive:
            self.job_alive = False
            self.done = not self.deleted
            self.deleted = False
        return False


def fake_calc(sims):
    from types import SimpleNamespace as NS

    stage = NS(lam_windows=[NS(sims=sims)], virtual_queue=NS(update=lambda: None))
    return NS(legs=[NS(stages=[stage])], slurm_config=NS(queue_check_interval=0))


def test_run_production_hold(batch, monkeypatch):
    """Test that jobs deleted during a hold are repeated without using up the
    attempts, and that nothing is submitted while the batch is on hold."""
    cfg = abfe.merge(
        abfe.DEFAULTS,
        {"submit_delay": 0, "protocol": {"runtime": 1.0, "max_sim_retries": 0}},
    )
    log = []
    sims = [FakeSim(log), FakeSim(log)]

    # Without a hold a lost simulation fails the ligand once the attempts are used
    sims[0].deleted = True
    with pytest.raises(RuntimeError, match="1 simulation"):
        abfe.run_production(fake_calc(sims), cfg, lambda done, total: None, batch)
    assert log == ["run", "run"]

    # The batch is put on hold after the first submission, and the job is deleted
    log.clear()
    sims = [FakeSim(log), FakeSim(log)]
    sims[0].deleted = True
    original_run = FakeSim.run

    def run_then_hold(self, runtime):
        original_run(self, runtime)
        if log == ["run"]:
            open(abfe.hold_path(batch), "w").close()

    def release(batch_dir):
        log.append("waited")
        os.remove(abfe.hold_path(batch_dir))
        return True

    monkeypatch.setattr(sims[0], "run", lambda runtime: run_then_hold(sims[0], runtime))
    monkeypatch.setattr(
        abfe, "wait_while_held", lambda d: abfe.is_held(d) and release(d)
    )
    progress = []
    abfe.run_production(
        fake_calc(sims), cfg, lambda done, total: progress.append(done), batch
    )
    # The second simulation was only submitted after the release
    assert log == ["run", "waited", "run", "run"]
    assert all(sim.done for sim in sims)
    assert progress[-1] == 2 and max(progress) == 2
