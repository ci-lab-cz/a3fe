"""Unit and regression tests for the _virtual_queue module."""

import getpass
import logging
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from ..configuration import JobStatus, PbsConfig, SlurmConfig
from ..run._virtual_queue import Job, VirtualQueue


def test_job():
    """Test that the Job class works correctly"""
    job = Job(1, ["echo", "hello"])
    assert job.virtual_job_id == 1
    assert job.command_list == ["echo", "hello"]
    assert job.slurm_job_id is None
    job.slurm_job_id = 1234
    assert job.slurm_job_id == 1234


def test_virtual_queue():
    """Check that the virtual queue works correctly. Note that we
    can't test submit, kill, or update as these require a slurm queue to
     submit to, which we don't have on the CI server."""
    with TemporaryDirectory() as dirname:
        # Set up the virtual queue and mess around with queues
        v_queue = VirtualQueue(queue_len_lim=30, log_dir=dirname)
        # Manually do most of submit
        virtual_job_id = v_queue._available_virt_job_id
        # Increment this so that it is never used again for this queue
        v_queue._available_virt_job_id += 1
        job1 = Job(virtual_job_id, ["echo", "hello"])
        v_queue._pre_queue.append(job1)
        v_queue._logger.info(f"{job1} submitted")
        # Add a job straight to the slurm queue
        job2 = Job(virtual_job_id, ["echo", "hello"])
        v_queue._slurm_queue.append(job2)
        v_queue._update_log()

        # Check that the total queue is the combination of the slurm and prequeue
        assert v_queue.queue == [job1, job2]
        # Check that logging is working
        assert os.path.isfile(os.path.join(dirname, "virtual_queue.log"))

        # Check that flushing works correctly.
        # Flusing pays no attention to the status of the jobs
        v_queue._flush()
        assert v_queue.queue == []

        # Check that we can change the stream log level of the stream handler
        v_queue.stream_log_level = logging.DEBUG
        assert v_queue._logger.handlers[0].level == logging.DEBUG


class _MockScheduler:
    """Mock the executables of a real scheduler (SLURM or PBS) by patching
    the subprocess functions used by the virtual queue and scheduler configs."""

    def __init__(self, config_cls):
        self.config_cls = config_cls
        self.active_ids = []
        self.next_id = 100
        self.calls = []
        self.queue_reads = 0

    def which(self, executable):
        """Only the executables of the mocked scheduler are present."""
        slurm_executables = ["sbatch", "squeue", "scancel"]
        pbs_executables = ["qsub", "qselect", "qdel"]
        if self.config_cls is SlurmConfig:
            return executable if executable in slurm_executables else None
        return executable if executable in pbs_executables else None

    def run(self, cmd, **kwargs):
        self.calls.append(cmd)
        result = type("Result", (), {"returncode": 0, "stderr": ""})()
        if cmd[0] == "qselect":
            self.queue_reads += 1
            assert cmd == PbsConfig.get_queue_cmd(getpass.getuser())
            result.stdout = "".join(f"{i}.server\n" for i in self.active_ids)
        elif cmd[0] in ("scancel", "qdel"):
            self.active_ids.remove(int(cmd[1]))
        else:
            raise AssertionError(f"Unexpected command {cmd}")
        return result

    def popen(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[0] in ("squeue", "awk", "grep", "paste"):
            if cmd[0] == "squeue":
                self.queue_reads += 1
            # The SLURM queue is read with a pipeline of commands
            assert self.config_cls is SlurmConfig
            queue = ",".join(str(i) for i in self.active_ids).encode()
            return type(
                "Process",
                (),
                {"stdout": None, "communicate": lambda self: (queue, None)},
            )()
        assert cmd[0] == ("qsub" if self.config_cls is PbsConfig else "sbatch")
        job_id = self.next_id
        self.next_id += 1
        self.active_ids.append(job_id)
        if self.config_cls is PbsConfig:
            output = f"{job_id}.server\n"
        else:
            output = f"Submitted batch job {job_id}\n"
        process = type("Process", (), {})()
        process.stdout = type("Stdout", (), {"read": lambda self: output.encode()})()
        return process


@pytest.mark.parametrize("config_cls", [SlurmConfig, PbsConfig])
def test_virtual_queue_submit_update_kill(config_cls):
    """Check that jobs are submitted, tracked, and killed correctly with
    each scheduler, using a mocked real queue."""
    scheduler = _MockScheduler(config_cls)
    with (
        TemporaryDirectory() as dirname,
        patch("subprocess.run", side_effect=scheduler.run),
        patch("subprocess.Popen", side_effect=scheduler.popen),
        patch("shutil.which", side_effect=scheduler.which),
    ):
        config = config_cls(output="job.out")
        file_base = config.get_slurm_output_file_base(dirname)
        cmd_list = config.get_submission_cmds("echo hello", dirname)
        with open(os.path.join(dirname, "job.out"), "w") as f:
            f.write("All fine\n")

        v_queue = VirtualQueue(queue_len_lim=2, log_dir=dirname)
        # With no jobs, the SLURM queue is still read (as it always was),
        # but nothing is run on a PBS-only cluster
        v_queue.update()
        assert scheduler.queue_reads == (1 if config_cls is SlurmConfig else 0)

        jobs = [v_queue.submit(cmd_list, file_base) for _ in range(3)]
        # Only two jobs fit in the real queue
        assert [job.slurm_job_id for job in jobs] == [100, 101, None]
        assert v_queue._slurm_queue == jobs[:2]
        assert v_queue._pre_queue == jobs[2:]
        assert all(job.status == JobStatus.QUEUED for job in jobs)

        # Finish the first job - the third should be submitted
        scheduler.active_ids.remove(100)
        v_queue.update()
        assert jobs[0].status == JobStatus.FINISHED
        assert jobs[2].slurm_job_id == 102
        assert v_queue.queue == jobs[1:]

        # Kill the second job
        v_queue.kill(jobs[1])
        kill_executable = "qdel" if config_cls is PbsConfig else "scancel"
        assert scheduler.calls[-1] == [kill_executable, "101"]
        assert jobs[1].status == JobStatus.KILLED
        assert v_queue.queue == jobs[2:]

        # Fail the third job
        with open(os.path.join(dirname, "job.out"), "w") as f:
            f.write("Particle coordinate is NaN\n")
        scheduler.active_ids.remove(102)
        v_queue.update()
        assert jobs[2].status == JobStatus.FAILED
        assert v_queue.queue == []
