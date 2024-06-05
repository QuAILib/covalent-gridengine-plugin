# Copyright 2021 Agnostiq Inc.
# Copyright 2024 National Institute of Advanced Industrial Science and Technology.
#
# This file is part of Covalent.
#
# Licensed under the Apache License 2.0 (the "License"). A copy of the
# License may be obtained with this software package or at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Use of this file is prohibited except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grid Engine executor plugin for the Covalent dispatcher."""

import asyncio
import os
import re
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import aiofiles
import asyncssh
import cloudpickle as pickle
import defusedxml.ElementTree
from aiofiles import os as async_os
from covalent._results_manager.result import Result
from covalent._shared_files import logger
from covalent._shared_files.config import get_config
from covalent._shared_files.exceptions import TaskCancelledError, TaskRuntimeError
from covalent.executor.executor_plugins.remote_executor import RemoteExecutor

app_log = logger.app_log
log_stack_info = logger.log_stack_info

executor_plugin_name = "GridEngineExecutor"

_EXECUTOR_PLUGIN_DEFAULTS = {
    "username": "",
    "address": "",
    "port": 22,
    "ssh_key_file": "",
    "bashrc_path": "$HOME/.bashrc",
    "qsub_args": {},
    "embedded_qsub_args": {},
    "cleanup": True,
    "remote_workdir": "covalent-workdir",
    "create_unique_workdir": False,
    "cache_dir": str(Path(get_config("dispatcher.cache_dir")).expanduser().resolve()),
    "poll_freq": 60,
    "remote_cache": ".cache/covalent",
    "log_stdout": "stdout.log",
    "log_stderr": "stderr.log",
}


class _JobState(Enum):
    """Enum for the state of job submitted to Grid Engine.

    Reference: https://manpages.ubuntu.com/manpages/jammy/en/man5/sge_status.5.html
    """

    NOT_IN_QUEUE = auto()
    ZOMBIE = auto()
    ERROR = auto()
    IN_QUEUE = auto()
    UNKNOWON = auto()


class GridEngineExecutor(RemoteExecutor):
    """Grid Engine executor plugin class.

    Args:
        username: Username used to authenticate over SSH (i.e. what you use to login to `address`).
        address: Remote address or hostname of the Grid Engine login node.
        port: Remote port of the Grid Engine login node.
        ssh_key_file: Private RSA key used to authenticate over SSH (usually at ~/.ssh/id_rsa).
        cert_file: Certificate file used to authenticate over SSH, if required (usually has extension .pub).
        passphrase: passphrase used to decrypt Private RSA key when loading them, if required.
            Note that hard coding your passphrase is not advised.
        bashrc_path: Path to the bashrc file to source before running the function.
        prerun_commands: List of shell commands to run before running the pickled function.
        postrun_commands: List of shell commands to run after running the pickled function.
        qsub_args: Dictionary of args used when executing command `qsub` on remote machine.
        embedded_qsub_args: Dictionary of args embedded into the submit script.
        cleanup: Whether to perform cleanup or not on remote machine.
        remote_workdir: Working directory on the remote cluster.
        create_unique_workdir: Whether to create a unique working (sub)directory for each task.
        cache_dir: Local cache directory used by this executor for temporary files.
        poll_freq: Frequency with which to poll a submitted job. Always is >= 30.
        remote_cache: Remote server cache directory used for temporary files.
        log_stdout: The path to the file to be used for redirecting stdout.
        log_stderr: The path to the file to be used for redirecting stderr.
        time_limit: time limit for the task.
        retries: Number of times to retry execution upon failure.
    """

    def __init__(
        self,
        # SSH credentials
        username: str | None = None,
        address: str | None = None,
        port: int | None = None,
        ssh_key_file: str | None = None,
        cert_file: str | None = None,
        passphrase: str | None = None,
        # executor parameters
        bashrc_path: str | None = None,
        prerun_commands: list[str] | None = None,
        postrun_commands: list[str] | None = None,
        qsub_args: dict[str, str | list[str]] | None = None,
        embedded_qsub_args: dict[str, str | list[str]] | None = None,
        cleanup: bool | None = None,
        # Covalent parameters
        remote_workdir: str | None = None,
        create_unique_workdir: bool | None = None,
        cache_dir: str | None = None,
        # RemoteExecutor parameters
        poll_freq: int | None = None,
        remote_cache: str = "",
        log_stdout: str = "",
        log_stderr: str = "",
        time_limit: int = -1,
        retries: int = 0,
        *args,
        **kwargs,
    ) -> None:
        poll_freq = poll_freq or get_config("executors.gridengine.poll_freq")

        if poll_freq is not None and poll_freq < 30:
            print("Polling frequency will be increased to 30 seconds.")
            poll_freq = 30

        remote_cache = remote_cache or get_config("executors.gridengine.remote_cache")
        log_stdout = log_stdout or get_config("executors.gridengine.log_stdout")
        log_stderr = log_stderr or get_config("executors.gridengine.log_stderr")

        super().__init__(
            poll_freq=poll_freq,
            remote_cache=remote_cache,
            log_stdout=log_stdout,
            log_stderr=log_stderr,
            time_limit=time_limit,
            retries=retries,
        )

        # SSH credentials
        self.username = username or get_config("executors.gridengine.username")
        self.address = address or get_config("executors.gridengine.address")
        self.port = port or get_config("executors.gridengine.port")

        self.ssh_key_file = ssh_key_file or get_config("executors.gridengine.ssh_key_file")
        self.ssh_key_file = str(Path(self.ssh_key_file).expanduser().resolve())

        try:
            self.cert_file = cert_file or get_config("executors.gridengine.cert_file")
            self.cert_file = str(Path(self.cert_file).expanduser().resolve())
        except KeyError:
            self.cert_file = None

        try:
            self.passphrase = passphrase or get_config("executors.gridengine.passphrase")
        except KeyError:
            self.passphrase = None

        # Covalent parameters
        self.remote_workdir = remote_workdir or get_config("executors.gridengine.remote_workdir")
        self.create_unique_workdir = (
            create_unique_workdir
            if create_unique_workdir is not None
            else get_config("executors.gridengine.create_unique_workdir")
        )

        self.cache_dir = cache_dir or get_config("executors.gridengine.cache_dir")
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())

        # Make sure local cache dir exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # executor parameters
        try:
            self.bashrc_path = bashrc_path or get_config("executors.gridengine.bashrc_path")
        except KeyError:
            self.bashrc_path = None

        try:
            self.prerun_commands = (
                list(prerun_commands)
                if prerun_commands
                else get_config("executors.gridengine.prerun_commands")
            )
        except KeyError:
            self.prerun_commands = []

        try:
            self.postrun_commands = (
                list(postrun_commands)
                if postrun_commands
                else get_config("executors.gridengine.postrun_commands")
            )
        except KeyError:
            self.postrun_commands = []

        # To allow passing empty dictionary
        if qsub_args is None:
            qsub_args = get_config("executors.gridengine.qsub_args")
        self.qsub_args = deepcopy(qsub_args)

        # To allow passing empty dictionary
        if embedded_qsub_args is None:
            embedded_qsub_args = get_config("executors.gridengine.embedded_qsub_args")
        self.embedded_qsub_args = deepcopy(embedded_qsub_args)

        self.cleanup = (
            cleanup if cleanup is not None else get_config("executors.gridengine.cleanup")
        )

    async def _validate_credentials(self) -> bool:
        """Validates the credentials (username, address, ssh key file and certification file) required to establish SSH connections.

        Args:
            None

        Returns:
            boolean indicating if the str parameters are non-empty strings and the file parameters exist.

        Raises:
            ValueError: If one of required parameters is None or empty string.
            FileNotFoundError: If the file does not exist.
        """
        if not self.username:
            raise ValueError("username is a required parameter in the Grid Engine plugin.")

        if not self.address:
            raise ValueError("address is a required parameter in the Grid Engine plugin.")

        if not self.ssh_key_file:
            raise ValueError("ssh_key_file is a required parameter in the Grid Engine plugin.")

        if not Path(self.ssh_key_file).is_file():
            raise FileNotFoundError(f"SSH key file {self.ssh_key_file} does not exist.")

        if (self.cert_file is not None) and (not Path(self.cert_file).is_file()):
            raise FileNotFoundError(
                f"Certificate file {self.cert_file} is assigned but does not exist."
            )

        return True

    async def _client_connect(self) -> asyncssh.SSHClientConnection:
        """Helper function for connecting to the remote host through asyncssh module.

        Args:
            None

        Returns:
            The connection object

        Raises:
            ValueError: If `username`, `address` or `ssh_key_file` is None or empty string.
            RuntimeError: If SSH connection could not be established.
        """

        await self._validate_credentials()

        ssh_key_file_path = Path(self.ssh_key_file).expanduser().resolve()

        if self.passphrase is not None:
            private_key = asyncssh.read_private_key(ssh_key_file_path, passphrase=self.passphrase)
        else:
            private_key = asyncssh.read_private_key(ssh_key_file_path)

        if self.cert_file:
            cert_file_path = Path(self.cert_file).expanduser().resolve()
            client_keys = [
                (
                    private_key,
                    asyncssh.read_certificate(cert_file_path),
                )
            ]
        else:
            client_keys = [private_key]

        try:
            conn = await asyncssh.connect(
                host=self.address,
                port=self.port,
                username=self.username,
                client_keys=client_keys,
                known_hosts=None,
            )

        except Exception as e:
            raise RuntimeError(
                f"Could not connect to host: '{self.address}' port: '{self.port}' as user: '{self.username}'",
                e,
            )

        return conn

    def _format_py_script(self, func_filename: str, result_filename: str) -> str:
        """Create the Python script that executes the pickled python function.

        Args:
            func_filename: Name of the pickled function.
            result_filename: Name of the pickled result.

        Returns:
            script: String object containing a script executes the pickled python function.
        """

        func_filename = os.path.join(self.remote_workdir, func_filename)
        result_filename = os.path.join(self.remote_workdir, result_filename)
        return f"""
import os
from pathlib import Path

import cloudpickle as pickle

with open(Path(os.path.expandvars("{func_filename}")).expanduser().resolve(), "rb") as f:
    function, args, kwargs = pickle.load(f)

result = None
exception = None

try:
    result = function(*args, **kwargs)
except Exception as e:
    exception = e

with open(Path(os.path.expandvars("{result_filename}")).expanduser().resolve(), "wb") as f:
    pickle.dump((result, exception), f)
"""

    def _format_submit_script(
        self, python_version: str, py_script_filename: str, current_remote_workdir: str
    ) -> str:
        """Create the shell script that defines the job, that runs the python script.

        Args:
            python_version: Python version required by the pickled function.
            py_script_filename: Name of the python script.
            current_remote_workdir: Current working directory on the remote machine.

        Returns:
            script: String object containing a script.
        """

        # preamble
        preamble_lines = []
        shebang = "#!/bin/bash\n"
        preamble_lines.append(shebang)
        for key, value in self.embedded_qsub_args.items():
            if type(value) is list:
                for arg_value in value:
                    embedded_qsub_arg_str = f"#$ -{key}" + (f" {arg_value}" if arg_value else "")
                    preamble_lines.append(embedded_qsub_arg_str)
            else:
                embedded_qsub_arg_str = f"#$ -{key}" + (f" {value}" if value else "")
                preamble_lines.append(embedded_qsub_arg_str)
        preamble_lines.append("")
        preamble_str = "\n".join(preamble_lines)

        # Source commands
        if self.bashrc_path:
            source_text = f"source {self.bashrc_path}\n"
        else:
            source_text = ""

        # chdir to current working directory
        cd_workdir_command = f"cd {current_remote_workdir}"

        # runs pre-run commands
        if self.prerun_commands:
            prerun_commands_str = "\n".join([""] + self.prerun_commands + [""])
        else:
            prerun_commands_str = ""

        # checks remote python version
        remote_python_version = f"""
remote_py_version=$(python3 -c "print('.'.join(map(str, __import__('sys').version_info[:2])))")
if [[ "{python_version}" != $remote_py_version ]] ; then
  >&2 echo "Python version mismatch. Please install Python {python_version} in the compute environment."
  exit 199
fi
"""

        remote_py_filename = os.path.join(current_remote_workdir, py_script_filename)
        python_cmd = f"python3 {remote_py_filename}"

        # runs post-run commands
        if self.postrun_commands:
            postrun_commands_str = "\n".join([""] + self.postrun_commands + [""])
        else:
            postrun_commands_str = ""

        # assemble commands into script body
        # check remote python version after pre-run commands
        script_body = "\n".join(
            [prerun_commands_str, remote_python_version, python_cmd, postrun_commands_str, "wait"]
        )

        # assemble script
        return "".join([preamble_str, source_text, cd_workdir_command, script_body])

    async def _upload_task(
        self,
        conn: asyncssh.SSHClientConnection,
        func_filename: str,
        py_script_filename: str,
        submit_script_filename: str,
        current_remote_workdir: str,
    ) -> None:
        """Upload the required files to the remote machine using SCP.

        Args:
            conn: Connection object to connect the the remote machine
            func_filename: Path to the function file to be uploaded.
            py_script_filename: Path to the python script runs the function.
            submit_script_filename: Path to the script that will be submitted to Grid Engine.
            current_remote_workdir: Current working directory on the remote machine. files uploaded under this directory.

        Returns:
            None
        """

        await asyncssh.scp(func_filename, (conn, current_remote_workdir))
        await asyncssh.scp(py_script_filename, (conn, current_remote_workdir))
        await asyncssh.scp(submit_script_filename, (conn, current_remote_workdir))

    async def submit_task(
        self,
        conn: asyncssh.SSHClientConnection,
        remote_submit_script_filename: str,
    ) -> asyncssh.SSHCompletedProcess:
        """Submit the task to Grid Engine with `qsub` and return the corresponding output.

        Args:
            conn: Connection object to connect to the remote machine.
            remote_submit_script_filename: Path to the remote submit script file.

        Returns:
            asyncssh.SSHCompletedProcess: Containing information about the output of executing `qsub`.
        """

        qsub_args_list = []
        for key, value in self.qsub_args.items():
            if type(value) is list:
                for arg_value in value:
                    qsub_args_list.append(f"-{key}" + (f" {arg_value}" if arg_value else ""))
            else:
                qsub_args_list.append(f"-{key}" + (f" {value}" if value else ""))

        qsub_args_str = " ".join(qsub_args_list)

        submit_command = "qsub"
        if qsub_args_str:
            submit_command += f" {qsub_args_str}"
        submit_command += f" {remote_submit_script_filename}"

        return await conn.run(submit_command)

    def _parse_qstat(self, qstat_string: str) -> dict[str, str]:
        """Helper function that parse the output of `qstat -xml` into a dictionary with Grid Engine job ID as key and state of the job as value.

        Args:
            qstat_string: String of the output of executing `qstat -xml`.

        Returns:
            job_dict: A dictionary with Grid Engine job ID as key and state of the job as value.

        Raises:
            RuntimeError: If failed to parse the xml_string.
        """

        job_dict = dict()

        try:
            tree = defusedxml.ElementTree.fromstring(qstat_string)
        except Exception as err:
            raise RuntimeError("Failed to parse the result of executing `qstat -xml`.") from err
        for job in tree.iter("job_list"):
            job_id = job.find("JB_job_number").text
            status = job.find("state").text

            job_dict[job_id] = status

        return job_dict

    def _parse_qacct(self, qacct_string: str) -> dict[str, str]:
        """Helper function that parse the output of `qacct -j job_id` and return as a dictionary.

        Args:
            qacct_string: String of the output of executing `qacct -j job_id`.

        Returns:
            qacct_dict: A dictionary of the output of `qacct -j job_id`.

        Raises:
            RuntimeError: If failed to parse a line of the qacct_string.
        """

        lines = qacct_string.split("\n")
        qacct_dict = dict()
        for line in lines:
            if line[:3] == "===":
                # skip the header line
                continue

            parsed_result = re.match(r"(?P<key>\S+)(\s+)(?P<value>.*)", line)
            if parsed_result is None:
                raise RuntimeError("Failed to parse a line of stdout of executing qacct.")

            key = str.strip(parsed_result.group("key"))
            value = str.strip(parsed_result.group("value"))
            qacct_dict[key] = value

        return qacct_dict

    async def get_status(self, conn: asyncssh.SSHClientConnection, job_id: str) -> str | None:
        """Query the status of a job previously submitted to Grid Engine with executing `qstat -u {username} -xml`.

        Args:
            conn: SSH connection object.
            job_id: Grid Engine job ID.

        Returns:
            job_state: A string describing the job state. Return None if the job not in queue.

        Raises:
            RuntimeError: If failed to execute `qstat -u {username} -xml` on the remote machine.
        """

        qstat_cmd = f"qstat -u {self.username} -xml"
        qstat_proc = await conn.run(qstat_cmd)

        if qstat_proc.returncode != 0:
            raise RuntimeError(
                f"returncode: {qstat_proc.returncode}, stderr: {str(qstat_proc.stderr)}"
            )

        job_dict = self._parse_qstat(str(qstat_proc.stdout).strip())

        return job_dict.get(job_id)

    def _classify_job_state(self, raw_job_state: str | None) -> _JobState:
        """Helper function to classify the job state.

        Args:
            raw_job_state: String describing the state of the job obtained from the stdout of executing `qstat`.

        Returns:
            job_state: One of _JobState describing the job state. Return None if the job not in queue.
        """

        if raw_job_state is None:
            return _JobState.NOT_IN_QUEUE
        elif raw_job_state == "z":
            return _JobState.ZOMBIE
        elif "E" in raw_job_state:
            return _JobState.ERROR
        elif re.search(r"qw|r|t|d|R|T|w|h|S|s", raw_job_state):
            return _JobState.IN_QUEUE
        else:
            return _JobState.UNKNOWON

    async def _poll_task(self, conn: asyncssh.SSHClientConnection, job_id: str) -> None:
        """Poll a Grid Engine job until completion.

        Args:
            conn: SSH connection object.
            job_id: Grid Engine job ID.

        Returns:
            None

        Raises:
            RuntimeError: If the state of the job is Error or if either `exit_status` or `failed` of the job is not 0.
            TaskCancelledError: If the job has been requested to be cancelled.
        """

        raw_job_state = await self.get_status(conn, job_id)
        job_state = self._classify_job_state(raw_job_state)

        while job_state == _JobState.IN_QUEUE:
            await asyncio.sleep(self.poll_freq)
            raw_job_state = await self.get_status(conn, job_id)
            job_state = self._classify_job_state(raw_job_state)
        if job_state == _JobState.UNKNOWON:
            raise RuntimeError(f"job ID {job_id} become to an unknown state.")
        elif job_state == _JobState.ERROR:
            raise RuntimeError(f"job ID {job_id} become to an error state.")

        # Verify if the job is canceled.
        if await self.get_cancel_requested():
            raise TaskCancelledError(f"Batch job with Job ID: {job_id} requested to be cancelled")

        await asyncio.sleep(self.poll_freq)
        qacct_cmd = f"qacct -j {job_id}"
        qacct_proc = await conn.run(qacct_cmd)

        if qacct_proc.returncode != 0:
            raise RuntimeError(
                f"failed to execute `qacct -j {job_id}`. returncode: {qacct_proc.returncode}, stderr: {str(qacct_proc.stderr)}"
            )

        job_info = self._parse_qacct(str(qacct_proc.stdout).strip())

        failed_code = int(job_info["failed"])
        exit_status = int(job_info["exit_status"])

        if failed_code != 0 or exit_status != 0:
            raise RuntimeError(
                f"Job ID {job_id} failed. failed' of qacct: {failed_code}, 'exit_status' of qacct: {exit_status}."
            )

    async def query_result(
        self,
        conn: asyncssh.SSHClientConnection,
        remote_result_filename: str,
        remote_stdout_filename: str,
        remote_stderr_filename: str,
        task_results_dir: str,
    ) -> tuple[Result, bytes, bytes, Exception]:
        """Query and retrieve the task result including stdout and stderr logs.

        Args:
            conn: SSH connection object.
            remote_result_filename: Name of the pickled result file on the remote_machine.
            remote_stdout_filename: Name of the log file of stdout on the remote machine.
            remote_stderr_filename: Name of the log file of stderr on the remote machine.
            task_results_dir: Directory on the Covalent server where the result will be copied.
        Returns:
            result: Task result.
            stdout: stdout log.
            stderr: stderr log.
            exception: Exception raised during task execution.
        """

        # Check the result file exists on the remote backend
        proc = await conn.run(f"test -e {remote_result_filename}")
        if proc.returncode != 0:
            raise FileNotFoundError(
                proc.returncode, str(proc.stderr).strip(), remote_result_filename
            )

        # Copy result file from remote machine to Covalent server
        local_result_filename = os.path.join(
            task_results_dir, os.path.basename(remote_result_filename)
        )
        await asyncssh.scp((conn, remote_result_filename), local_result_filename)

        # Copy stdout, stderr from remote machine to Covalent server
        local_stdout_file = os.path.join(
            task_results_dir, os.path.basename(remote_stdout_filename)
        )
        local_stderr_file = os.path.join(
            task_results_dir, os.path.basename(remote_stderr_filename)
        )

        await asyncssh.scp((conn, remote_stdout_filename), local_stdout_file)
        await asyncssh.scp((conn, remote_stderr_filename), local_stderr_file)

        # Unpickle the result file and delete the file from Covalent server
        async with aiofiles.open(local_result_filename, "rb") as f:
            contents = await f.read()
            result, exception = pickle.loads(contents)

        # Read the stdout and stderr and delete files from Covalent server
        async with aiofiles.open(local_stdout_file, "r") as f:
            stdout = await f.read()

        async with aiofiles.open(local_stderr_file, "r") as f:
            stderr = await f.read()

        await async_os.remove(local_result_filename)
        await async_os.remove(local_stdout_file)

        # If the job was submitted using the option `-j y`, local_stderr_file will be the same as local_stdout_file.
        # So in this case, trying to delete local_stderr_file will result in an exception.
        if local_stderr_file != local_stdout_file:
            await async_os.remove(local_stderr_file)

        return result, stdout, stderr, exception

    def _make_remote_log_filenames(self, job_id: str, job_script_filename: str) -> tuple[str, str]:
        """Helper function that creates paths for stdout and stderr files on the remote machine.
        These paths are determined by the settings of the `-N`, `-o`, `e`, and `-j` options.


            Args:
                job_id: Grid Engine job ID.
                job_script_filename: Name of the remote submit script file.

            Returns:
                remote_stdout_filename: Name of the log file of stdout on the remote machine.
                remote_stderr_filename: Name of the log file of stderr on the remote machine.
        """

        # Determine the job name from the name of the job script and the value of the option `N`.
        job_name = job_script_filename
        if "N" in self.qsub_args or "N" in self.embedded_qsub_args:
            job_name = self.qsub_args.get("N") or self.embedded_qsub_args.get("N")

        default_stdout_filename = f"{job_name}.o{job_id}"
        default_stderr_filename = f"{job_name}.e{job_id}"

        def _make_remote_filename(specified_filename: str | None, default_filename: str) -> str:
            """Inner function that make the string of the path to file on the remote machine."""

            remote_filename = specified_filename

            if not remote_filename:
                # If the filename is not specified, log file will be output to home directory with default name.
                return default_filename
            elif remote_filename[-1] == "/":
                # If remote_filename is a path of a directory, log file will be output with default name.
                remote_filename = os.path.join(remote_filename, default_filename)

            return remote_filename

        stdout_filename = self.qsub_args.get("o") or self.embedded_qsub_args.get("o")
        remote_stdout_filename = _make_remote_filename(stdout_filename, default_stdout_filename)

        stderr_filename = self.qsub_args.get("e") or self.embedded_qsub_args.get("e")
        remote_stderr_filename = _make_remote_filename(stderr_filename, default_stderr_filename)

        # If an option `-j y[es]` exist, stderr will merge with stdout.
        if "j" in self.qsub_args or "j" in self.embedded_qsub_args:
            j_option_value = self.qsub_args.get("j") or self.embedded_qsub_args.get("j")
            if j_option_value in {"y", "yes"}:
                remote_stderr_filename = remote_stdout_filename

        return remote_stdout_filename, remote_stderr_filename

    async def run(
        self, function: Callable, args: list, kwargs: dict, task_metadata: dict
    ) -> Result:
        """Run a function on a remote machine using Grid Engine.

        Args:
            function: Function to be executed.
            args: List of positional arguments to be passed to the function.
            kwargs: Dictionary of keyword arguments to be passed to the function.
            task_metadata: Dictionary of metadata associated with the task.

        Returns:
            result: Result object containing the result of the function execution.
        """

        dispatch_id = task_metadata["dispatch_id"]
        node_id = task_metadata["node_id"]
        results_dir = task_metadata["results_dir"]
        task_results_dir = os.path.join(results_dir, dispatch_id)

        if self.create_unique_workdir:
            current_remote_workdir = os.path.join(
                self.remote_workdir, dispatch_id, "node_" + str(node_id)
            )
        else:
            current_remote_workdir = self.remote_workdir

        # Specify file names.
        result_filename = f"result-{dispatch_id}-{node_id}.pkl"
        job_script_filename = f"jobscript-{dispatch_id}-{node_id}.sh"
        py_script_filename = f"script-{dispatch_id}-{node_id}.py"
        func_filename = f"func-{dispatch_id}-{node_id}.pkl"

        result = None

        if await self.get_cancel_requested():
            app_log.debug("Task {dispatch_id}-{node_id} has been cancelled don't proceed")
            raise TaskCancelledError(f"Task {dispatch_id}-{node_id} requested to be cancelled.")

        async with await self._client_connect() as conn:
            py_version_func = ".".join(function.args[0].python_version.split(".")[:2])
            app_log.debug(f"Python version: {py_version_func}")

            if await self.get_cancel_requested():
                app_log.debug("Task {dispatch_id}-{node_id} has been cancelled don't proceed")
                raise TaskCancelledError(
                    f"Task {dispatch_id}-{node_id} requested to be cancelled."
                )

            # Create the remote directory
            app_log.debug(f"Creating remote work directory {current_remote_workdir} ...")
            cmd_mkdir_remote = f"mkdir -p {current_remote_workdir}"
            proc_mkdir_remote = await conn.run(cmd_mkdir_remote)

            if proc_mkdir_remote.returncode != 0:
                raise RuntimeError(str(proc_mkdir_remote.stderr).strip())

            async with aiofiles.tempfile.NamedTemporaryFile(
                dir=self.cache_dir, delete=False
            ) as temp_func_file:
                # Pickle the function and write to file
                app_log.debug("Writing pickled function, args, kwargs to file...")
                await temp_func_file.write(pickle.dumps((function, args, kwargs)))
                await temp_func_file.flush()
                local_func_filename = os.path.join(
                    os.path.dirname(temp_func_file.name), func_filename
                )
                os.rename(temp_func_file.name, local_func_filename)

            async with aiofiles.tempfile.NamedTemporaryFile(
                dir=self.cache_dir, mode="w", delete=False
            ) as temp_py_script_file:
                # Format the function execution script and write to file
                python_exec_script = self._format_py_script(func_filename, result_filename)
                app_log.debug("Writing python run-function script to tempfile...")
                await temp_py_script_file.write(python_exec_script)
                await temp_py_script_file.flush()
                local_py_script_filename = os.path.join(
                    os.path.dirname(temp_py_script_file.name), py_script_filename
                )
                os.rename(temp_py_script_file.name, local_py_script_filename)

            async with aiofiles.tempfile.NamedTemporaryFile(
                dir=self.cache_dir, mode="w", delete=False
            ) as temp_job_script_file:
                # Format the job script and write to file
                submit_script = self._format_submit_script(
                    py_version_func, py_script_filename, current_remote_workdir
                )
                app_log.debug("Writing Grid Engine submit script to tempfile...")
                await temp_job_script_file.write(submit_script)
                await temp_job_script_file.flush()
                local_job_script_filename = os.path.join(
                    os.path.dirname(temp_job_script_file.name), job_script_filename
                )
                os.rename(temp_job_script_file.name, local_job_script_filename)

            if await self.get_cancel_requested():
                app_log.debug("Task {dispatch_id}-{node_id} has been cancelled don't proceed")
                raise TaskCancelledError(
                    f"Task {dispatch_id}-{node_id} requested to be cancelled."
                )

            # Copy files to the remote machine
            await self._upload_task(
                conn,
                local_func_filename,
                local_py_script_filename,
                local_job_script_filename,
                current_remote_workdir,
            )

            remote_func_filename = os.path.join(current_remote_workdir, func_filename)
            remote_py_script_filename = os.path.join(current_remote_workdir, py_script_filename)
            remote_job_script_filename = os.path.join(current_remote_workdir, job_script_filename)

            if await self.get_cancel_requested():
                app_log.debug("Task {dispatch_id}-{node_id} has been cancelled don't proceed")
                raise TaskCancelledError(
                    f"Task {dispatch_id}-{node_id} requested to be cancelled."
                )

            proc_submit_task = await self.submit_task(conn, remote_job_script_filename)

            if proc_submit_task.returncode != 0:
                raise RuntimeError(str(proc_submit_task.stderr).strip())

            submit_task_stdout = str(proc_submit_task.stdout).strip()

            app_log.debug(f"Job submitted with stdout: {submit_task_stdout}")

            job_id = re.findall("[0-9]+", submit_task_stdout)[0]

            await self.set_job_handle(handle=job_id)

            # Identify the names of the stdout and stderr log files on the remote machine.
            remote_stdout_filename, remote_stderr_filename = self._make_remote_log_filenames(
                job_id, job_script_filename
            )

            app_log.debug(f"Polling Grid Engine with job_id: {job_id} ...")
            try:
                await self._poll_task(conn, job_id)
            except RuntimeError as err:
                app_log.debug(
                    f"Submitted Task with job_id: {job_id} failed with error: {str(err)}"
                )
                raise
            except TaskCancelledError:
                app_log.debug("Task has been cancelled don't proceed")
                raise

            app_log.debug(f"Querying result with job_id: {job_id} ...")
            remote_result_filename = os.path.join(current_remote_workdir, result_filename)
            result, stdout, stderr, exception = await self.query_result(
                conn,
                remote_result_filename,
                remote_stdout_filename,
                remote_stderr_filename,
                task_results_dir,
            )

            print(stdout, end="", file=self._task_stdout)
            print(stderr, end="", file=self._task_stderr)

            if exception:
                app_log.debug(f"An exception has occurred in the task {dispatch_id}-{node_id}:")
                app_log.debug(f"exception message: {exception}")
                print(exception, end="", file=self._task_stderr)
                raise TaskRuntimeError from exception

            app_log.debug("Preparing for teardown")
            self._local_func_filename = local_func_filename
            self._local_job_script_filename = local_job_script_filename
            self._local_py_script_filename = local_py_script_filename
            self._remote_func_filename = remote_func_filename
            self._remote_job_script_filename = remote_job_script_filename
            self._remote_py_script_filename = remote_py_script_filename
            self._remote_result_filename = remote_result_filename
            self._remote_stdout_filename = remote_stdout_filename
            self._remote_stderr_filename = remote_stderr_filename

        app_log.debug("SSH connection closed, returning result")

        return result

    async def cancel(self, task_metadata: dict, job_id: str | None) -> bool:
        """
        Cancel the job.

        Args:
            task_metadata: Dictionary with the task's dispatch_id and node id.
            job_id: Unique ID assigned to the job by the backend.

        Returns:
            True if cancel succeeded, False if failed.
        """

        if job_id is None:
            app_log.debug("cancel was called while the task was not yet run.")
            return True

        app_log.debug(f"Cancelling job ID {job_id}...")
        async with await self._client_connect() as conn:
            qdel_cmd = f"qdel {job_id}"
            proc = await conn.run(qdel_cmd)

            is_canceled = proc.returncode == 0

            if not is_canceled:
                stderr_str = str(proc.stderr).strip()
                app_log.debug(
                    f"Failed to cancel the job with task_metadata: {task_metadata}, job_handle: {job_id}. error: {stderr_str}"
                )

        app_log.debug("SSH connection closed, cancel complete")

        return is_canceled

    async def teardown(self, task_metadata: dict) -> None:
        """Perform cleanup on remote machine and Covalent server.
        If self.cleanup is False, teardown do nothing.

        Args:
            task_metadata: Dictionary of metadata associated with the task. This variable is always passed by Covalent, but is never used.

        Returns:
            None
        """
        if self.cleanup:
            app_log.debug("Performing cleanup on remote...")
            try:
                async with await self._client_connect() as conn:
                    await self._perform_cleanup(conn=conn)

            except Exception as err:
                app_log.warning(
                    "Grid Engine cleanup could not successfully complete. Nonfatal error."
                )
                app_log.warning(err)

            app_log.debug("SSH connection closed, teardown complete")

    async def _perform_cleanup(self, conn: asyncssh.SSHClientConnection) -> None:
        """Function to perform cleanup on remote machine and Covalent server.

        Args:
            conn: SSH connection object

        Returns:
            None
        """

        local_files_to_remove = [
            self._local_func_filename,
            self._local_job_script_filename,
            self._local_py_script_filename,
        ]

        remote_files_to_remove = [
            self._remote_func_filename,
            self._remote_job_script_filename,
            self._remote_py_script_filename,
            self._remote_result_filename,
            self._remote_stdout_filename,
            self._remote_stderr_filename,
        ]

        for local_file in local_files_to_remove:
            await async_os.remove(local_file)

        for remote_file in remote_files_to_remove:
            await conn.run(f"rm {remote_file}")
