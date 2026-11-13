"""Unit and regression tests for the PbsConfig class."""

import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from a3fe import PbsConfig, SlurmConfig


def test_create_default_config():
    """Test that the default config is created correctly."""
    config = PbsConfig()
    assert config.queue == "workq"
    assert config.ngpus == 1


def test_modify_config():
    """Check that we can modify the config."""
    config = PbsConfig()
    config.queue = "test_queue"
    with pytest.raises(ValueError):
        config.ncpus = 0


def test_config_pickle_and_load():
    """Test that the config can be pickled and loaded."""
    with TemporaryDirectory() as dirname:
        config = PbsConfig(ncpus=4, setup_commands=["module load gromacs"])
        config.dump(dirname)
        assert os.path.isfile(os.path.join(dirname, "pbs_config.yaml"))
        config2 = PbsConfig.load(dirname)
        assert config == config2


def test_get_select():
    """Test that the select statement is generated correctly."""
    assert PbsConfig().get_select() == "1:ncpus=1:ngpus=1"
    assert PbsConfig(ncpus=8, ngpus=0).get_select() == "1:ncpus=8"
    assert (
        PbsConfig(ncpus=8, ngpus=2, mem="16gb").get_select()
        == "1:ncpus=8:ngpus=2:mem=16gb"
    )
    assert (
        PbsConfig(ncpus=8, select="2:ncpus=4:mpiprocs=4").get_select()
        == "2:ncpus=4:mpiprocs=4"
    )


def test_get_submission_cmds():
    """
    Test that the submission commands are generated correctly
    and that the script is written correctly.
    """
    # Tmpdir to store the script
    with TemporaryDirectory() as dirname:
        config = PbsConfig(
            queue="test_queue",
            walltime="12:00:00",
            ncpus=4,
            ngpus=1,
            mem="8gb",
            output="test-%j.out",
            setup_commands=[],
        )
        submission_cmds = config.get_submission_cmds(
            cmd="echo 'Test'", run_dir=dirname, script_name="test"
        )
        script_path = os.path.join(dirname, "test.sh")
        assert submission_cmds == ["qsub", script_path]

        expected_script = (
            "#!/bin/bash\n"
            "#PBS -q test_queue\n"
            "#PBS -l walltime=12:00:00\n"
            "#PBS -l select=1:ncpus=4:ngpus=1:mem=8gb\n"
            "#PBS -j oe\n"
            "#PBS -R oe\n"
            "#PBS -V\n"
            f"\ncd {os.path.abspath(dirname)} || exit 1\n"
            'exec > "test-${PBS_JOBID%%.*}.out" 2>&1\n'
            "\necho 'Test'\n"
        )

        with open(script_path, "r") as f:
            script = f.read()

        assert script == expected_script


def test_get_submission_cmds_no_walltime():
    """Test that no time limit is requested if the walltime is None."""
    with TemporaryDirectory() as dirname:
        config = PbsConfig(walltime=None, setup_commands=[])
        config.get_submission_cmds(cmd="echo 'Test'", run_dir=dirname)
        with open(os.path.join(dirname, "a3fe.sh"), "r") as f:
            script = f.read()

        assert "walltime" not in script
        assert "#PBS -q workq\n#PBS -l select=1:ncpus=1:ngpus=1\n" in script


def test_get_submission_cmds_options():
    """
    Test that the script is written correctly with extra options,
    setup commands, and without exporting the environment.
    """
    with TemporaryDirectory() as dirname:
        config = PbsConfig(
            ngpus=0,
            export_env=False,
            extra_options={"A": "my_account", "l": "place=excl"},
            setup_commands=["module load gromacs", "export FOO=bar"],
        )
        _ = config.get_submission_cmds(
            cmd="echo 'Test'", run_dir=dirname, script_name="test"
        )
        with open(os.path.join(dirname, "test.sh"), "r") as f:
            script = f.read()

        assert "#PBS -l select=1:ncpus=1\n" in script
        assert "#PBS -V" not in script
        assert "#PBS -A my_account\n" in script
        assert "#PBS -l place=excl\n" in script
        assert script.endswith(
            "2>&1\nmodule load gromacs\nexport FOO=bar\n\necho 'Test'\n"
        )


def test_default_setup_commands(monkeypatch, tmp_path):
    """
    Test that the active conda environment (if any) is activated by default.
    """
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    assert PbsConfig().setup_commands == []

    conda_sh = tmp_path / "etc" / "profile.d" / "conda.sh"
    conda_sh.parent.mkdir(parents=True)
    conda_sh.touch()
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "bin" / "conda"))
    monkeypatch.setenv("CONDA_PREFIX", "/path/to/envs/a3fe")
    assert PbsConfig().setup_commands == [
        f"source {conda_sh}",
        "conda activate /path/to/envs/a3fe",
    ]
    # The default can be overridden
    assert PbsConfig(setup_commands=[]).setup_commands == []


def test_get_output_file_base():
    """
    Test that the output file base is generated correctly.
    """
    config = PbsConfig()
    assert config.get_slurm_output_file_base(run_dir="test_dir") == "test_dir/pbs-"
    config.output = "fixed_name.out"
    assert (
        config.get_slurm_output_file_base(run_dir="test_dir")
        == "test_dir/fixed_name.out"
    )


def test_get_default_partition():
    """
    Test that the default queue is found correctly.
    """
    with patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = (
            "Server: pbs-server\n    server_state = Active\n"
            "    default_queue = workq\n    log_events = 511\n"
        )
        assert PbsConfig.get_default_partition() == "workq"


def test_get_default_partition_retry():
    """
    Test that the query is repeated if the server can not be reached,
    and that an informative error is raised if it keeps failing.
    """
    failure = MagicMock(stdout="", stderr="qstat: cannot connect to server")
    success = MagicMock(stdout="    default_queue = workq\n", stderr="")
    with patch.object(PbsConfig, "_query_wait", 0):
        with patch("subprocess.run", side_effect=[failure, failure, success]):
            assert PbsConfig.get_default_partition() == "workq"
        with patch("subprocess.run", return_value=failure) as mocked_run:
            with pytest.raises(ValueError, match="cannot connect to server"):
                PbsConfig.get_default_partition()
            assert mocked_run.call_count == PbsConfig._query_attempts


@pytest.mark.parametrize(
    "pbsnodes_output, expected_ngpus",
    [
        ("node1\n     resources_available.ncpus = 352\n", 0),
        ("node1\n     resources_available.ngpus = 0\n", 0),
        (
            "node1\n     resources_available.ngpus = 0\n"
            "node2\n     resources_available.ngpus = 4\n",
            1,
        ),
    ],
)
def test_get_default_config(pbsnodes_output, expected_ngpus):
    """
    Test that the default config requests a GPU only if there are any.
    """

    def mocked_run(cmd, **kwargs):
        result = type("Result", (), {})()
        result.stdout = (
            "    default_queue = my_queue\n" if cmd[0] == "qstat" else pbsnodes_output
        )
        return result

    with patch("subprocess.run", side_effect=mocked_run):
        config = PbsConfig.get_default_config()
    assert config.queue == "my_queue"
    assert config.ngpus == expected_ngpus


def test_scheduler_commands():
    """Test the commands used to communicate with PBS, and the parsing of
    their output."""
    assert PbsConfig.get_queue_cmd("me") == ["qselect", "-u", "me", "-s", "QRHWTSBU"]
    assert PbsConfig.get_kill_cmd(1234) == ["qdel", "1234"]
    assert PbsConfig.parse_job_id("1234.pbs-server\n") == 1234
    assert PbsConfig.parse_job_id("1234") == 1234
    for bad_output in ["", "qsub: Unknown queue", "qsub: Job exceeds limit 12"]:
        with pytest.raises(ValueError):
            PbsConfig.parse_job_id(bad_output)
    # Array jobs should be ignored
    queue = "6952.pbs-server\n6953.pbs-server\n6954[].pbs-server\n6955[3].pbs-server\n"
    assert PbsConfig.parse_queue(queue) == [6952, 6953]
    assert PbsConfig.parse_queue("") == []


def test_read_queue():
    """Test that the queue is read correctly and that failures raise an error."""
    with patch("subprocess.run") as mocked_run:
        mocked_run.return_value.returncode = 0
        mocked_run.return_value.stdout = "12.server\n13.server\n"
        assert PbsConfig.read_queue("me") == [12, 13]

        mocked_run.return_value.returncode = 1
        mocked_run.return_value.stderr = "Connection refused"
        with pytest.raises(ValueError, match="Connection refused"):
            PbsConfig.read_queue("me")


def test_is_submission_cmd():
    """Test that PBS submission commands are recognised."""
    with TemporaryDirectory() as dirname:
        assert PbsConfig.is_submission_cmd(
            PbsConfig().get_submission_cmds("ls", dirname)
        )
        assert not PbsConfig.is_submission_cmd(
            SlurmConfig().get_submission_cmds("ls", dirname)
        )
    assert PbsConfig.is_submission_cmd(["/opt/pbs/bin/qsub", "a3fe.sh"])
    assert not PbsConfig.is_submission_cmd(["echo", "hello"])
    assert not PbsConfig.is_submission_cmd([])


@pytest.mark.parametrize(
    "available, expected",
    [(["sbatch"], False), (["qsub"], True), (["sbatch", "qsub"], False), ([], False)],
)
def test_is_available(available, expected):
    """Test that PBS is only used by default if SLURM is not present."""
    with patch(
        "shutil.which", side_effect=lambda exe: exe if exe in available else None
    ):
        assert PbsConfig.is_available() == expected
