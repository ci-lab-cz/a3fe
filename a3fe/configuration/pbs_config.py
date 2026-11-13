"""Configuration classes for PBS (PBS Professional / OpenPBS) configuration."""

__all__ = [
    "PbsConfig",
]

import os as _os
import re as _re
import shlex as _shlex
import shutil as _shutil
import subprocess as _subprocess
import time as _time
from typing import ClassVar as _ClassVar
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional

import yaml as _yaml
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict as _ConfigDict
from pydantic import Field as _Field


def _get_default_setup_commands() -> _List[str]:
    """
    Get the commands which activate the currently active conda environment
    (if any). Unlike SLURM, PBS starts each job in a new login shell, where
    the user's profile typically resets the environment (e.g. by activating
    the base conda environment), even if the environment is exported.
    """
    conda_prefix = _os.environ.get("CONDA_PREFIX")
    conda_exe = _os.environ.get("CONDA_EXE")
    if not conda_prefix or not conda_exe:
        return []
    conda_sh = _os.path.join(
        _os.path.dirname(_os.path.dirname(conda_exe)), "etc", "profile.d", "conda.sh"
    )
    if not _os.path.isfile(conda_sh):
        return []
    return [
        f"source {_shlex.quote(conda_sh)}",
        f"conda activate {_shlex.quote(conda_prefix)}",
    ]


class PbsConfig(_BaseModel):
    """
    Pydantic model for holding a PBS configuration. This targets
    PBS Professional / OpenPBS (i.e. the "select" resource syntax).

    A PbsConfig can be supplied wherever a SlurmConfig is accepted, e.g.
    ``a3fe.Calculation(slurm_config=a3fe.PbsConfig(queue="gpu"))``.
    """

    # The executable used to submit jobs. This is the first element of the lists
    # returned by get_submission_cmds, and is used to identify PBS jobs.
    submit_executable: _ClassVar[str] = "qsub"

    # Job states which mean that the job has not finished yet: queued, running,
    # held, waiting, in transit, suspended, begun (arrays) and suspended due
    # to workstation activity. Exiting (E) jobs are deliberately excluded, as
    # the job script has finished by then and PBS can take minutes to clean up.
    _active_states: _ClassVar[str] = "QRHWTSBU"

    # Number of attempts to query the PBS server, which can be temporarily
    # unreachable (e.g. when it is busy), and the time to wait between them.
    _query_attempts: _ClassVar[int] = 5
    _query_wait: _ClassVar[float] = 10

    queue: str = _Field("workq", description="PBS queue to submit to.")
    walltime: _Optional[str] = _Field(
        "24:00:00",
        description="Time limit for the PBS job. If None, no time limit is "
        "requested and the default of the queue applies.",
    )
    ncpus: int = _Field(1, ge=1, description="Number of CPU cores to request.")
    ngpus: int = _Field(
        1,
        ge=0,
        description="Number of GPUs to request - normally one. Set to 0 when "
        "running on CPU-only nodes.",
    )
    mem: _Optional[str] = _Field(
        None, description="Memory to request, e.g. '8gb'. Not requested if None."
    )
    select: _Optional[str] = _Field(
        None,
        description="Full select statement, e.g. '1:ncpus=8:ngpus=1:scratch_local=10gb'. "
        "If supplied, this is used instead of the one built from ncpus, ngpus and mem.",
    )
    output: str = _Field(
        "pbs-%j.out",
        description="Output file for the PBS job, written to the run directory "
        "while the job is running. '%j' is replaced by the job id.",
    )
    export_env: bool = _Field(
        True,
        description="Whether to export the environment of the submitting process "
        "to the job (qsub -V), as SLURM does by default. This makes e.g. the "
        "active conda environment available to the job.",
    )
    setup_commands: _List[str] = _Field(
        default_factory=_get_default_setup_commands,
        description="Commands to run in the job script before the main command, "
        "e.g. ['module load gromacs']. PBS starts jobs in a new login shell, so "
        "these should set up the environment required by the job. By default, "
        "the conda environment which is active when the configuration is created "
        "(if any) is activated.",
    )

    extra_options: _Dict[str, str] = _Field(
        {},
        description="Extra options to pass to PBS. For example, {'A': 'my_account'} "
        "is written as '#PBS -A my_account'.",
    )
    queue_check_interval: int = _Field(
        30, ge=1, description="Interval in seconds between PBS queue status checks."
    )
    job_submission_wait: int = _Field(
        300, ge=1, description="Wait time in seconds for job submission to PBS queue."
    )

    model_config = _ConfigDict(validate_assignment=True)

    def get_select(self) -> str:
        """Get the select statement describing the requested resources."""
        if self.select:
            return self.select
        select = f"1:ncpus={self.ncpus}"
        if self.ngpus:
            select += f":ngpus={self.ngpus}"
        if self.mem:
            select += f":mem={self.mem}"
        return select

    def get_submission_cmds(
        self, cmd: str, run_dir: str, script_name: str = "a3fe"
    ) -> _List[str]:
        """
        Generates the PBS submission commands list based on the configuration.

        Parameters
        ----------
        cmd : str
            Command to run during the PBS job.

        run_dir : str
            Directory to run the PBS job in.

        script_name : str, optional, default="a3fe"
            Name of the script file to write. Note that when running many jobs from the
            same directory, this should be unique to avoid overwriting the script file.

        Returns
        -------
        List[str]
            The list of PBS arguments.
        """
        # First, write the script to a file
        script_path = _os.path.join(run_dir, f"{script_name}.sh")

        script = (
            "#!/bin/bash\n"
            f"#PBS -q {self.queue}\n"
            + (f"#PBS -l walltime={self.walltime}\n" if self.walltime else "")
            + f"#PBS -l select={self.get_select()}\n"
            # We write the output file ourselves (see below), so merge and
            # remove the (empty) standard output and error files written by PBS
            "#PBS -j oe\n"
            "#PBS -R oe\n"
        )
        if self.export_env:
            script += "#PBS -V\n"

        for key, value in self.extra_options.items():
            script += f"#PBS -{key} {value}\n"

        # PBS starts jobs in the home directory and only copies the output
        # to its destination once the job has finished. Move to the run directory
        # and write the output there directly, so that it can be read while
        # the job is running (as with SLURM).
        output = self.output.replace("%j", "${PBS_JOBID%%.*}")
        script += (
            f"\ncd {_shlex.quote(_os.path.abspath(run_dir))} || exit 1\n"
            f'exec > "{output}" 2>&1\n'
        )

        for setup_command in self.setup_commands:
            script += f"{setup_command}\n"

        script += f"\n{cmd}\n"

        with open(script_path, "w") as f:
            f.write(script)

        return ["qsub", script_path]

    def get_slurm_output_file_base(self, run_dir: str) -> str:
        """
        Get the base name of the PBS output file. (This has the same name
        as the SlurmConfig method so that the configurations are interchangeable.)

        Parameters
        ----------
        run_dir : str
            Directory the job was submitted from.

        Returns
        -------
        str
            The base name of the PBS output file.
        """
        return run_dir + "/" + self.output.split("%")[0]

    @classmethod
    def is_submission_cmd(cls, command_list: _List[str]) -> bool:
        """
        Check whether a command list (as returned by get_submission_cmds)
        submits a job to PBS.
        """
        return (
            len(command_list) > 0
            and _os.path.basename(command_list[0]) == cls.submit_executable
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check whether PBS, and not SLURM, is available on this machine."""
        return _shutil.which("sbatch") is None and _shutil.which("qsub") is not None

    @classmethod
    def get_default_partition(cls) -> str:
        """
        Get the default PBS queue.

        Raises
        ------
        ValueError
            If the default queue could not be found, e.g. because the server
            could not be reached.
        """
        for attempt in range(cls._query_attempts):
            if attempt > 0:
                _time.sleep(cls._query_wait)
            qstat = _subprocess.run(
                ["qstat", "-Bf"],
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
                text=True,
            )
            match = _re.search(r"default_queue = (\S+)", qstat.stdout)
            if match is not None:
                return match.group(1)
        raise ValueError(f"Could not find the default PBS queue: {qstat.stderr}")

    @classmethod
    def has_gpus(cls) -> bool:
        """Check whether any of the PBS nodes offers GPUs."""
        pbsnodes = _subprocess.run(
            ["pbsnodes", "-a"], stdout=_subprocess.PIPE, text=True
        )
        ngpus = _re.findall(r"resources_available\.ngpus = (\d+)", pbsnodes.stdout)
        return any(int(n) > 0 for n in ngpus)

    @classmethod
    def get_default_config(cls) -> "PbsConfig":
        """
        Get a configuration which uses the default PBS queue, and which
        requests a GPU only if the cluster has any.
        """
        return cls(queue=cls.get_default_partition(), ngpus=1 if cls.has_gpus() else 0)

    @classmethod
    def get_queue_cmd(cls, user: str) -> _List[str]:
        """Get the command which lists the active PBS jobs of the user."""
        return ["qselect", "-u", user, "-s", cls._active_states]

    @classmethod
    def parse_queue(cls, output: str) -> _List[int]:
        """
        Extract the ids of the active jobs from the output of qselect.
        Array jobs are ignored.
        """
        # Job ids are of the form "1234.server" ("1234[].server" for arrays)
        return [
            int(job_id.split(".")[0]) for job_id in output.split() if "[" not in job_id
        ]

    @classmethod
    def read_queue(cls, user: str) -> _List[int]:
        """
        Get the ids of all active (e.g. queued or running) PBS jobs of the user.

        Raises
        ------
        ValueError
            If the queue could not be read, e.g. because the server is busy.
        """
        result = _subprocess.run(
            cls.get_queue_cmd(user),
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Could not read the PBS queue: {result.stderr.strip()}")
        return cls.parse_queue(result.stdout)

    @classmethod
    def parse_job_id(cls, submission_output: str) -> int:
        """Extract the job id from the output of qsub."""
        # The last line of the output is of the form "1234.server"
        lines = submission_output.strip().splitlines()
        match = _re.fullmatch(r"(\d+)(\.\S*)?", lines[-1].strip()) if lines else None
        if match is None:
            raise ValueError(f"Could not find a PBS job id in '{submission_output}'")
        return int(match.group(1))

    @classmethod
    def get_kill_cmd(cls, job_id: int) -> _List[str]:
        """Get the command which cancels a PBS job."""
        return ["qdel", str(job_id)]

    def dump(self, save_dir: str) -> None:
        """
        Dumps the configuration to a YAML file.

        Parameters
        ----------
        save_dir : str
            Directory to save the YAML file to.
        """
        model_dict = self.model_dump()

        save_path = save_dir + "/" + self.get_file_name()
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, load_dir: str) -> "PbsConfig":
        """
        Loads the configuration from a YAML file.

        Parameters
        ----------
        load_dir : str
            Directory to load the YAML file from.

        Returns
        -------
        PbsConfig
            The loaded configuration.
        """
        with open(load_dir + "/" + cls.get_file_name(), "r") as f:
            model_dict = _yaml.safe_load(f)
        return cls(**model_dict)

    @staticmethod
    def get_file_name() -> str:
        """
        Get the name of the PBS configuration file.
        """
        return "pbs_config.yaml"
