"""Unit tests for the a3fe-abfe command line tool. No scheduler is required."""

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
            ["#PBS -N drv.lig_B_2", "#PBS -l select=1:ncpus=2:mem=16gb", "#PBS -A acc"],
            "walltime",
        ),
        (
            "slurm",
            "21;cluster\n",
            ["#SBATCH --job-name=drv.lig_B_2", "#SBATCH --mem=16gb", "#SBATCH --A=acc"],
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
    config = abfe.make_scheduler_config(cfg, "lig_A")
    assert isinstance(config, PbsConfig)
    assert (config.queue, config.walltime) == ("long", None)
    assert config.get_select() == "1:ncpus=8:mem=8gb"
    assert config.extra_options == {"A": "acc", "N": "sim.lig_A"}
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
    config = abfe.make_scheduler_config(cfg, "lig_A")
    assert isinstance(config, SlurmConfig)
    assert (config.partition, config.time, config.gres) == ("cpu", "01:00:00", "none")
    assert config.extra_options == {
        "job-name": "sim.lig_A",
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
    set_state(batch, "lig_B_2", state=abfe.DONE, sims_done=4, sims_total=4)
    set_state(batch, "lig_lig_A", state=abfe.RUNNING, job_id="11", stage="production")
    jobs = {"11": ("drv.lig_lig_A", "R"), "12": ("sim.lig_lig_A", "Q")}
    with patch.object(abfe.Scheduler, "active_jobs", return_value=jobs):
        abfe.main(["-d", batch, "status"])
    lines = capsys.readouterr().out.splitlines()
    assert any(
        line.split()
        == ["lig_B_2", "done", "-", "4/4", "0/0", "-4.5", "+/-", "1.2", "0m"]
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
