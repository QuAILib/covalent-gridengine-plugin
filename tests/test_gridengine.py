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

"""Tests for the Grid Engine executor plugin."""

import asyncio
import io
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from unittest import mock

import pytest
from covalent._shared_files.config import get_config, set_config
from covalent._shared_files.exceptions import TaskCancelledError, TaskRuntimeError
from covalent._workflow import TransportableObject

from covalent_gridengine_plugin.gridengine import GridEngineExecutor, _JobState

DATA_DIR = Path(os.path.dirname(__file__)) / "data"


def mock_key_read(*args, **kwargs):
    """Mock for asyncssh.read_private_key() and asyncssh.read_certificate()"""
    return True


def mock_basic_async(*args, **kwargs):
    """A basic async mock"""
    future = asyncio.Future()
    future.set_result(True)
    return future


def mock_wrapper_fn(function: TransportableObject, *args, **kwargs):
    """Minimum mocking of wrapper_fn"""

    fn = function.get_deserialized()

    output = fn(*args, **kwargs)

    return TransportableObject(output)


@pytest.fixture
def conn_mock():
    return mock.MagicMock()


@pytest.fixture
def proc_mock():
    return mock.MagicMock()


def test_init_defaults() -> None:
    """Test that initialization properly sets member variables in case of default values."""

    executor = GridEngineExecutor()

    assert executor.username == ""
    assert executor.address == ""
    assert executor.port == 22
    assert executor.ssh_key_file == str(Path("").expanduser().resolve())
    assert executor.cert_file is None
    assert executor.passphrase is None
    assert executor.bashrc_path == "$HOME/.bashrc"
    assert executor.prerun_commands == []
    assert executor.postrun_commands == []
    assert executor.qsub_args == {}
    assert executor.embedded_qsub_args == {}
    assert executor.poll_freq == 60
    assert executor.cleanup is True
    assert executor.remote_workdir == "covalent-workdir"
    assert executor.create_unique_workdir is False
    assert executor.cache_dir == str(
        Path(get_config("dispatcher.cache_dir")).expanduser().resolve()
    )
    assert executor.log_stdout == "stdout.log"
    assert executor.log_stderr == "stderr.log"
    assert executor.time_limit == -1
    assert executor.retries == 0


def test_init_poll_freq_auto_raised():
    """Test that poll_freq is auto-raised"""

    poll_freq = 10
    executor = GridEngineExecutor(poll_freq=poll_freq)

    assert executor.poll_freq == 30


def test_init_nondefaults():
    """Test that initialization properly sets member variables in case of non-default values."""

    args = {
        "username": "username",
        "address": "host",
        "port": 10022,
        "ssh_key_file": "~/.ssh/id_rsa",
        "cert_file": "~/.ssh/id_rsa.pub",
        "passphrase": "passphrase",
        "bashrc_path": "path/to/.bashrc",
        "prerun_commands": ["module load module1", "module load module2"],
        "postrun_commands": ["qacct -j 1"],
        "qsub_args": {
            "g": "groupname",
        },
        "embedded_qsub_args": {
            "l": ["arch=xxx", "h_rt=1:00:00"],
            "soft": "",
        },
        "poll_freq": 45,
        "cleanup": False,
        "remote_workdir": "path/to/remote_workdir",
        "create_unique_workdir": True,
        "cache_dir": "path/to/cache_dir",
        "log_stdout": "path/to/log_stdout.log",
        "log_stderr": "path/to/log_stderr.log",
        "time_limit": 60,
        "retries": 3,
    }
    executor = GridEngineExecutor(**args)

    assert executor.username == args["username"]
    assert executor.address == args["address"]
    assert executor.port == args["port"]
    assert executor.ssh_key_file == str(Path(args["ssh_key_file"]).expanduser().resolve())
    assert executor.cert_file == str(Path(args["cert_file"]).expanduser().resolve())
    assert executor.passphrase == args["passphrase"]
    assert executor.bashrc_path == args["bashrc_path"]
    assert executor.prerun_commands == args["prerun_commands"]
    assert executor.postrun_commands == args["postrun_commands"]
    assert executor.qsub_args == args["qsub_args"]
    assert executor.embedded_qsub_args == args["embedded_qsub_args"]
    assert executor.poll_freq == args["poll_freq"]
    assert executor.cleanup == args["cleanup"]
    assert executor.remote_workdir == args["remote_workdir"]
    assert executor.create_unique_workdir == args["create_unique_workdir"]
    assert executor.cache_dir == str(Path(args["cache_dir"]).expanduser().resolve())
    assert executor.log_stdout == args["log_stdout"]
    assert executor.log_stderr == args["log_stderr"]
    assert executor.time_limit == args["time_limit"]
    assert executor.retries == args["retries"]


def test_init_config_bashrc_path_not_exist():
    """Test that initialization is succeeded if config of bashrc_path is not exist and the arg is None."""

    start_config = deepcopy(get_config())
    config = get_config()
    config["executors"]["gridengine"].pop("bashrc_path", None)
    set_config(config)
    executor = GridEngineExecutor(
        username="username",
        address="host",
        ssh_key_file="~/.ssh/id_rsa",
    )
    assert not executor.__dict__["bashrc_path"]
    set_config(start_config)


def test_format_py_script():
    """Test that the python script (in string form) which is to be executed
    on the remote server is created with no errors."""

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        remote_workdir="/federation/test_user/.cache/covalent",
        cache_dir="~/.cache/covalent",
    )

    dispatch_id = "378cdd10-8443-11ee-99e2-a35cb4664dfd"
    task_id = 1
    func_filename = f"func-{dispatch_id}-{task_id}.pkl"
    result_filename = f"result-{dispatch_id}-{task_id}.pkl"

    try:
        py_script_str = executor._format_py_script(func_filename, result_filename)
        print(py_script_str)
    except Exception as exc:
        assert False, f"Exception while running _format_py_script: {exc}"
    assert func_filename in py_script_str
    assert result_filename in py_script_str


def test_format_submit_script_default():
    """Test that the shell script (in string form) which is to be submitted on
    the remote server is created with no errors."""

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        cache_dir="~/.cache/covalent",
    )

    python_version = ".".join(map(str, __import__("sys").version_info[:2]))
    remote_python_version_check_str = f"""
remote_py_version=$(python3 -c "print('.'.join(map(str, __import__('sys').version_info[:2])))")
if [[ "{python_version}" != $remote_py_version ]] ; then
  >&2 echo "Python version mismatch. Please install Python {python_version} in the compute environment."
  exit 199
fi
"""

    dispatch_id = "378cdd10-8443-11ee-99e2-a35cb4664dfd"
    task_id = 1
    py_filename = f"script-{dispatch_id}-{task_id}.py"

    try:
        submit_script_str = executor._format_submit_script(
            python_version, py_filename, remote_workdir
        )
        print(submit_script_str)
    except Exception as exc:
        assert False, f"Exception while running _format_submit_script with default options: {exc}"
    assert python_version in submit_script_str

    shebang = "#!/bin/bash\n"
    assert submit_script_str.startswith(shebang), f"Missing '{shebang[:-1]}' in the shell script"
    assert "source $HOME/.bashrc\n" in submit_script_str
    assert f"cd {remote_workdir}\n" in submit_script_str
    assert remote_python_version_check_str in submit_script_str
    assert f"python3 {os.path.join(remote_workdir, py_filename)}" in submit_script_str
    assert submit_script_str.endswith("wait"), "Missing 'wait' in the shell script"


def test_format_submit_script_without_bashrc_path():
    """Test that the shell script (in string form) which is to be submitted on
    the remote server is created with no errors."""

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        cache_dir="~/.cache/covalent",
    )
    executor.bashrc_path = ""

    python_version = ".".join(map(str, __import__("sys").version_info[:2]))

    dispatch_id = "378cdd10-8443-11ee-99e2-a35cb4664dfd"
    task_id = 1
    py_filename = f"script-{dispatch_id}-{task_id}.py"

    try:
        submit_script_str = executor._format_submit_script(
            python_version, py_filename, remote_workdir
        )
        print(submit_script_str)
    except Exception as exc:
        assert False, f"Exception while running _format_submit_script with default options: {exc}"
    assert python_version in submit_script_str

    shebang = "#!/bin/bash\n"
    assert submit_script_str.startswith(shebang), f"Missing '{shebang[:-1]}' in the shell script"
    assert "source" not in submit_script_str
    assert f"cd {remote_workdir}\n" in submit_script_str
    assert f"python3 {os.path.join(remote_workdir, py_filename)}" in submit_script_str
    assert submit_script_str.endswith("wait"), "Missing 'wait' in the shell script"


def test_format_submit_script_with_additional_commands():
    """Test that the shell script (in string form) which is to be submitted on
    the remote server is created with no errors."""

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        prerun_commands=["prerun1", "prerun2"],
        postrun_commands=["postrun1", "postrun2", "postrun3"],
        remote_workdir=remote_workdir,
        cache_dir="~/.cache/covalent",
    )

    python_version = ".".join(map(str, __import__("sys").version_info[:2]))
    remote_python_version_check_str = f"""
remote_py_version=$(python3 -c "print('.'.join(map(str, __import__('sys').version_info[:2])))")
if [[ "{python_version}" != $remote_py_version ]] ; then
  >&2 echo "Python version mismatch. Please install Python {python_version} in the compute environment."
  exit 199
fi
"""

    dispatch_id = "378cdd10-8443-11ee-99e2-a35cb4664dfd"
    task_id = 1
    py_filename = f"script-{dispatch_id}-{task_id}.py"

    try:
        submit_script_str = executor._format_submit_script(
            python_version, py_filename, remote_workdir
        )
        print(submit_script_str)
    except Exception as exc:
        assert False, f"Exception while running _format_submit_script with default options: {exc}"
    assert python_version in submit_script_str

    shebang = "#!/bin/bash\n"
    assert submit_script_str.startswith(shebang), f"Missing '{shebang[:-1]}' in the shell script"
    assert "source $HOME/.bashrc\n" in submit_script_str
    assert f"cd {remote_workdir}\n" in submit_script_str

    # check the script contains prerun commands and porstrun_commands, also check the order of commands
    assert (
        "\n".join(
            [
                "\n".join(executor.prerun_commands),
                "",
                remote_python_version_check_str,
                f"python3 {os.path.join(remote_workdir, py_filename)}",
                "",
                "\n".join(executor.postrun_commands),
                "",
            ]
        )
        in submit_script_str
    )
    assert submit_script_str.endswith("wait"), "Missing 'wait' in the shell script"


def test_format_submit_script_with_embedded_qsub_args():
    """Test that the shell script (in string form) which is to be submitted on
    the remote server is created with no errors."""

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        embedded_qsub_args={"l": ["arch=xxx", "h_rt=1:00:00"], "soft": ""},
        remote_workdir=remote_workdir,
        cache_dir="~/.cache/covalent",
    )

    python_version = ".".join(map(str, __import__("sys").version_info[:2]))

    dispatch_id = "378cdd10-8443-11ee-99e2-a35cb4664dfd"
    task_id = 1
    py_filename = f"script-{dispatch_id}-{task_id}.py"

    try:
        submit_script_str = executor._format_submit_script(
            python_version, py_filename, remote_workdir
        )
        print(submit_script_str)
    except Exception as exc:
        assert False, f"Exception while running _format_submit_script with default options: {exc}"
    assert python_version in submit_script_str

    shebang = "#!/bin/bash\n"
    assert submit_script_str.startswith(shebang), f"Missing '{shebang[:-1]}' in the shell script"

    embedded_args_list = ["#$ -l arch=xxx\n", "#$ -l h_rt=1:00:00\n", "#$ -soft\n"]
    for embedded_arg_str in embedded_args_list:
        assert embedded_arg_str in submit_script_str

    assert "source $HOME/.bashrc\n" in submit_script_str
    assert f"cd {remote_workdir}\n" in submit_script_str

    assert f"python3 {os.path.join(remote_workdir, py_filename)}" in submit_script_str
    assert submit_script_str.endswith("wait"), "Missing 'wait' in the shell script"


@pytest.mark.asyncio
async def test_upload_task(mocker, conn_mock):
    """Test that required_files for the task are uploaded to the remote machine with no errors."""

    mocker.patch("asyncssh.scp", return_value=mock.AsyncMock())

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
    )

    local_workdir = "home/user/local_workdir/"
    func_filename = local_workdir + "func-test.pkl"
    py_script_filename = local_workdir + "script-test.py"
    submit_script_filename = local_workdir + "submit-script-test.sh"

    await executor._upload_task(
        conn_mock, func_filename, py_script_filename, submit_script_filename, remote_workdir
    )


@pytest.mark.asyncio
async def test_submit_task(conn_mock, proc_mock):
    """Test that the command is executed with no error."""

    async def mock_run_succeeded(cmd):
        proc_mock.cmd = cmd
        return proc_mock

    conn_mock.run = mock_run_succeeded

    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        qsub_args={"l": ["arch=xxx", "h_rt=1:00:00"], "soft": ""},
    )

    remote_submit_script_filename = f"{remote_workdir}/submit-script-test.sh"

    qsub_args_str = "-l arch=xxx -l h_rt=1:00:00 -soft"
    expected_cmd = f"qsub {qsub_args_str} {remote_submit_script_filename}"
    proc = await executor.submit_task(conn_mock, remote_submit_script_filename)

    assert proc.cmd == expected_cmd, "executed command is not equal to the expected command"

    # without qsub_args
    executor.qsub_args = {}
    expected_cmd = f"qsub {remote_submit_script_filename}"
    proc = await executor.submit_task(conn_mock, remote_submit_script_filename)

    assert proc.cmd == expected_cmd, "executed command is not equal to the expected command"


@pytest.mark.asyncio
async def test_validate_credentials_no_exception(mocker):
    """Test for _validate_credentials raises no exception"""

    mocker.patch.object(Path, "is_file", return_value=True)

    executor = GridEngineExecutor(
        username="user", address="address", ssh_key_file="ssh_key_file", cert_file="cert_file"
    )
    await executor._validate_credentials()


@pytest.mark.asyncio
async def test_validate_credentials_raises_exception(mocker):
    """Test for _validate_credentials raises an exception"""

    # Test for ValueError
    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.username = None
        await executor._validate_credentials()
    assert str(err.value) == "username is a required parameter in the Grid Engine plugin."

    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.username = ""
        await executor._validate_credentials()
    assert str(err.value) == "username is a required parameter in the Grid Engine plugin."

    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.address = None
        await executor._validate_credentials()
    assert str(err.value) == "address is a required parameter in the Grid Engine plugin."

    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.address = ""
        await executor._validate_credentials()
    assert str(err.value) == "address is a required parameter in the Grid Engine plugin."

    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.ssh_key_file = None
        await executor._validate_credentials()
    assert str(err.value) == "ssh_key_file is a required parameter in the Grid Engine plugin."

    with pytest.raises(ValueError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        executor.ssh_key_file = ""
        await executor._validate_credentials()
    assert str(err.value) == "ssh_key_file is a required parameter in the Grid Engine plugin."

    # Test for FileNotFoundError
    not_exist_file_path = "file/does/not/exist"

    def mock_is_file(path_object: Path):
        if "file/does/not/exist" in str(path_object):
            return False
        return True

    mocker.patch.object(Path, "is_file", new=mock_is_file)

    with pytest.raises(FileNotFoundError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file=not_exist_file_path
        )
        await executor._validate_credentials()
    assert (
        str(err.value)
        == f"SSH key file {str(Path(not_exist_file_path).expanduser().resolve())} does not exist."
    )

    with pytest.raises(FileNotFoundError) as err:
        executor = GridEngineExecutor(
            username="user",
            address="address",
            ssh_key_file="ssh_key_file",
            cert_file=not_exist_file_path,
        )
        await executor._validate_credentials()
    assert (
        str(err.value)
        == f"Certificate file {str(Path(not_exist_file_path).expanduser().resolve())} is assigned but does not exist."
    )


@pytest.mark.asyncio
async def test_client_connect_no_exception(mocker):
    """Test for _client_connect with mocking .connect()"""

    mocker.patch("asyncssh.read_private_key", side_effect=mock_key_read)
    mocker.patch("asyncssh.read_certificate", side_effect=mock_key_read)
    mocker.patch("asyncssh.connect", side_effect=mock_basic_async)

    mocker.patch.object(GridEngineExecutor, "_validate_credentials", new=mock_basic_async)

    assert (
        await GridEngineExecutor(
            address="test_address", username="test_use", ssh_key_file="ssh_key_file"
        )._client_connect()
        is True
    )

    assert (
        await GridEngineExecutor(
            address="test_address",
            username="test_use",
            ssh_key_file="ssh_key_file",
            cert_file="cert_file",
        )._client_connect()
        is True
    )

    assert (
        await GridEngineExecutor(
            address="test_address",
            username="test_use",
            passphrase="passphrase",
            ssh_key_file="ssh_key_file",
            cert_file="cert_file",
        )._client_connect()
        is True
    )


@pytest.mark.asyncio
async def test_client_connect_raises_exception(mocker):
    """Test RuntimeError is raised when .connect() is failed with mocking .connect()"""

    mocker.patch("asyncssh.read_private_key", side_effect=mock_key_read)
    mocker.patch("asyncssh.read_certificate", side_effect=mock_key_read)
    mocked_error = Exception("mocked error")
    mocker.patch("asyncssh.connect", side_effect=mocked_error)

    mocker.patch.object(GridEngineExecutor, "_validate_credentials", new=mock_basic_async)

    with pytest.raises(RuntimeError) as err:
        executor = GridEngineExecutor(
            username="user", address="address", ssh_key_file="ssh_key_file"
        )
        await executor._client_connect()
    assert err.value.args == (
        f"Could not connect to host: '{executor.address}' port: '{executor.port}' as user: '{executor.username}'",
        mocked_error,
    )


def test_parse_qstat_no_error():
    """Test that parse the result of `qstat -xml` with no error."""

    with open(DATA_DIR / "sample_qstat_output.txt") as f:
        test_input = f.read()

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )
    parsed_result = executor._parse_qstat(test_input)

    assert len(parsed_result) == 3
    assert parsed_result["00000001"] == "r"
    assert parsed_result["00000002"] == "qw"
    assert parsed_result["00000003"] == "qw"


def test_parse_qstat_raise_error():
    """Test that parse_qstat raises error."""

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    with pytest.raises(RuntimeError):
        parsed_result = executor._parse_qstat("not xml")


def test_parse_qacct_no_error():
    """Test that parse the result of `qacct -j job_id` with no error."""

    with open(DATA_DIR / "sample_qacct_output.txt") as f:
        test_input = f.read()

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    parsed_result = executor._parse_qacct(test_input.strip())

    assert len(parsed_result) == 53
    assert parsed_result["qname"] == "gpu"
    assert parsed_result["hostname"] == "hostname"
    assert parsed_result["group"] == "groupname"
    assert parsed_result["owner"] == "username"
    assert parsed_result["project"] == "groupname"
    assert parsed_result["department"] == "groupname"
    assert parsed_result["jobname"] == "test.sh"
    assert parsed_result["jobnumber"] == "00000001"
    assert parsed_result["taskid"] == "undefined"
    assert parsed_result["pe_taskid"] == "NONE"
    assert parsed_result["account"] == "groupname"
    assert parsed_result["priority"] == "0"
    assert parsed_result["cwd"] == "NONE"
    assert parsed_result["submit_host"] == "submit_host"
    assert (
        parsed_result["submit_cmd"]
        == "/home/system/uge/latest/bin/lx-amd64/qsub -P groupname -l rt_F=1 test.sh"
    )
    assert parsed_result["qsub_time"] == "11/20/2023 14:59:21.269"
    assert parsed_result["start_time"] == "11/20/2023 14:59:40.791"
    assert parsed_result["end_time"] == "11/20/2023 14:59:50.826"
    assert parsed_result["granted_pe"] == "perack09"
    assert parsed_result["slots"] == "80"
    assert parsed_result["failed"] == "0"
    assert parsed_result["deleted_by"] == "NONE"
    assert parsed_result["exit_status"] == "199"
    assert parsed_result["ru_wallclock"] == "10.035"
    assert parsed_result["ru_utime"] == "0.018"
    assert parsed_result["ru_stime"] == "0.015"
    assert parsed_result["ru_maxrss"] == "8936"
    assert parsed_result["ru_ixrss"] == "0"
    assert parsed_result["ru_ismrss"] == "0"
    assert parsed_result["ru_idrss"] == "0"
    assert parsed_result["ru_isrss"] == "0"
    assert parsed_result["ru_minflt"] == "548"
    assert parsed_result["ru_majflt"] == "0"
    assert parsed_result["ru_nswap"] == "0"
    assert parsed_result["ru_inblock"] == "8"
    assert parsed_result["ru_oublock"] == "24"
    assert parsed_result["ru_msgsnd"] == "0"
    assert parsed_result["ru_msgrcv"] == "0"
    assert parsed_result["ru_nsignals"] == "0"
    assert parsed_result["ru_nvcsw"] == "14"
    assert parsed_result["ru_nivcsw"] == "1"
    assert parsed_result["wallclock"] == "51.494"
    assert parsed_result["cpu"] == "0.033"
    assert parsed_result["mem"] == "0.000"
    assert parsed_result["io"] == "0.001"
    assert parsed_result["iow"] == "0.000"
    assert parsed_result["ioops"] == "304"
    assert parsed_result["maxvmem"] == "19.578M"
    assert parsed_result["maxrss"] == "0.000"
    assert parsed_result["maxpss"] == "0.000"
    assert parsed_result["arid"] == "undefined"
    assert parsed_result["jc_name"] == "NONE"
    assert parsed_result["bound_cores"] == "NONE"


def test_parse_qacct_raise_error(mocker):
    """Test that _parse_qacct raises error."""

    mocker.patch("re.match", return_value=None)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    with pytest.raises(RuntimeError):
        parsed_result = executor._parse_qacct("test string")


@pytest.mark.asyncio
async def test_get_status(mocker, conn_mock, proc_mock):
    """Test that get_status works as expected."""

    proc_mock.returncode = 0

    async def mock_run(cmd):
        # memorize executed command in get_status
        proc_mock.cmd = cmd
        return proc_mock

    conn_mock.run = mock_run

    job_id1 = "00000001"
    expected1 = "r"
    job_id2 = "00000002"
    expected2 = "qw"
    job_id3 = "00000003"

    def _mock_parse_qstat(*_):
        return {job_id1: expected1, job_id2: expected2}

    mocker.patch.object(GridEngineExecutor, "_parse_qstat", new=_mock_parse_qstat)

    test_username = "test_user"
    executor = GridEngineExecutor(
        username=test_username,
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    actual_result1 = await executor.get_status(conn_mock, job_id1)
    actual_result2 = await executor.get_status(conn_mock, job_id2)
    actual_result3 = await executor.get_status(conn_mock, job_id3)

    assert proc_mock.cmd == f"qstat -u {test_username} -xml"
    assert actual_result1 == expected1
    assert actual_result2 == expected2
    assert actual_result3 is None


@pytest.mark.asyncio
async def test_get_status_raise_error(conn_mock, proc_mock):
    """Test that get_status raises error."""

    mock_returncode = 1
    mock_error_message = "mock error message"

    proc_mock.stdout = ""
    proc_mock.stderr = mock_error_message
    proc_mock.returncode = mock_returncode
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    job_id = "00000001"

    with pytest.raises(RuntimeError) as err:
        await executor.get_status(conn_mock, job_id)
    assert str(err.value) == f"returncode: {mock_returncode}, stderr: {mock_error_message}"


@pytest.mark.parametrize(
    "test_state, expected",
    [
        (None, _JobState.NOT_IN_QUEUE),
        ("dr", _JobState.IN_QUEUE),
        ("dT", _JobState.IN_QUEUE),
        ("dS", _JobState.IN_QUEUE),
        ("dS", _JobState.IN_QUEUE),
        ("ds", _JobState.IN_QUEUE),
        ("Eqw", _JobState.ERROR),
        ("Ehqw", _JobState.ERROR),
        ("z", _JobState.ZOMBIE),
        ("qw", _JobState.IN_QUEUE),
        ("hqw", _JobState.IN_QUEUE),
        ("Rq", _JobState.IN_QUEUE),
        ("s", _JobState.IN_QUEUE),
        ("ts", _JobState.IN_QUEUE),
        ("S", _JobState.IN_QUEUE),
        ("tS", _JobState.IN_QUEUE),
        ("T", _JobState.IN_QUEUE),
        ("tT", _JobState.IN_QUEUE),
        ("Rs", _JobState.IN_QUEUE),
        ("Rts", _JobState.IN_QUEUE),
        ("RS", _JobState.IN_QUEUE),
        ("RT", _JobState.IN_QUEUE),
        ("RtT", _JobState.IN_QUEUE),
        ("t", _JobState.IN_QUEUE),
        ("r", _JobState.IN_QUEUE),
        ("hr", _JobState.IN_QUEUE),
        ("Rr", _JobState.IN_QUEUE),
        ("u", _JobState.UNKNOWON),
    ],
)
def test_classify_job_state(test_state, expected):
    """Test that _classify_job_state return appropriate value corresponding to the job state."""
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    actual_job_state = executor._classify_job_state(test_state)
    assert actual_job_state == expected


@pytest.mark.asyncio
async def test_poll_task(mocker, conn_mock):
    """Test that _poll_task continues if the state of the job is a particular state and finish with no error."""

    patch_get_status = mocker.patch.object(
        GridEngineExecutor, "get_status", return_value=mock.AsyncMock()
    )
    mocker.patch.object(
        GridEngineExecutor,
        "_classify_job_state",
        side_effect=[
            _JobState.IN_QUEUE,
            _JobState.IN_QUEUE,
            _JobState.IN_QUEUE,
            _JobState.IN_QUEUE,
            _JobState.IN_QUEUE,
            _JobState.NOT_IN_QUEUE,
        ],
    )
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)
    mocker.patch.object(
        GridEngineExecutor, "_parse_qacct", return_value={"failed": "0", "exit_status": "0"}
    )

    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"

    async def mock_run(cmd):
        # memorize executed command in get_status
        proc_mock.cmd = cmd
        return proc_mock

    conn_mock.run = mock_run

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        poll_freq=30,
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    job_id = "00000001"
    await asyncio.wait_for(executor._poll_task(conn_mock, job_id), timeout=10)
    assert proc_mock.cmd == f"qacct -j {job_id}"
    assert patch_get_status.call_count == 6


@pytest.mark.parametrize(
    "test_state",
    [
        "u",
        "Eqw",
    ],
)
@pytest.mark.asyncio
async def test_poll_task_with_unknown_or_error_state(mocker, conn_mock, test_state):
    """Test that _poll_task raises an error if the state of the job is a particular state."""

    mocker.patch.object(GridEngineExecutor, "get_status", return_value=test_state)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        poll_freq=30,
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    with pytest.raises(RuntimeError):
        await executor._poll_task(conn_mock, "00000001")


@pytest.mark.asyncio
async def test_poll_task_failed_qacct(mocker, conn_mock, proc_mock):
    """Test that _poll_task raises an error when failed to execute qacct."""

    mocker.patch.object(GridEngineExecutor, "get_status", return_value=None)
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)

    proc_mock.returncode = 1
    proc_mock.stdout = ""
    proc_mock.stderr = "stderr"
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    executor = GridEngineExecutor(
        username="test_user", address="test_address", ssh_key_file="~/.ssh/id_rsa", poll_freq=30
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    with pytest.raises(RuntimeError):
        await executor._poll_task(conn_mock, "00000001")


@pytest.mark.parametrize(
    "failed_value, exit_status_value",
    [("0", "199"), ("1", "0"), ("1", "199")],
)
@pytest.mark.asyncio
async def test_poll_task_job_script_failed(
    mocker, conn_mock, proc_mock, failed_value, exit_status_value
):
    """Test that _poll_task raises an error when the job script is failed."""

    mocker.patch.object(GridEngineExecutor, "get_status", return_value=None)
    mocker.patch.object(
        GridEngineExecutor,
        "_parse_qacct",
        return_value={"failed": failed_value, "exit_status": exit_status_value},
    )
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)

    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        poll_freq=30,
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    with pytest.raises(RuntimeError):
        await executor._poll_task(conn_mock, "00000001")


@pytest.mark.asyncio
async def test_poll_task_task_cancelled(mocker, conn_mock):
    """Test that _poll_task raises TaskCancelledError if the task has been requested to be cancelled."""

    mocker.patch.object(GridEngineExecutor, "get_status", return_value=None)
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=True)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        poll_freq=30,
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    with pytest.raises(TaskCancelledError):
        await executor._poll_task(conn_mock, "00000001")


@pytest.mark.asyncio
async def test_poll_task_for_zombie_state(mocker, conn_mock):
    """Test that _poll_task finishes polling if the job state is _JobState.ZOMBIE."""

    patch_get_status = mocker.patch.object(
        GridEngineExecutor, "get_status", return_value=mock.AsyncMock()
    )
    mocker.patch.object(GridEngineExecutor, "_classify_job_state", return_value=_JobState.ZOMBIE)
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)
    mocker.patch.object(
        GridEngineExecutor, "_parse_qacct", return_value={"failed": "0", "exit_status": "0"}
    )

    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        poll_freq=30,
    )
    # To speed up of this test, set poll_freq to 1 specially.
    executor.poll_freq = 1

    await asyncio.wait_for(executor._poll_task(conn_mock, "00000001"), timeout=10)
    assert patch_get_status.call_count == 1


@pytest.mark.asyncio
async def test_query_result_no_error(mocker, conn_mock, proc_mock):
    """Test querying results works as expected with no error."""

    # Now mock result files.
    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    mocker.patch("asyncssh.scp", return_value=mock.AsyncMock())

    # Don't actually try to remove result files:
    mock_async_os_remove = mock.AsyncMock(return_value=None)
    patch_remove = mocker.patch("aiofiles.os.remove", side_effect=mock_async_os_remove)

    # Mock the opening of specific result files:
    expected_results = [1, 2, 3, 4, 5]
    expected_error = None
    expected_stdout = "output logs"
    expected_stderr = "output errors"
    mocker.patch("cloudpickle.loads", return_value=(expected_results, expected_error))

    current_remote_workdir = "/home/test_user/workdir"
    task_results_dir = "/path/to/results_dir"
    remote_result_filename = os.path.join(current_remote_workdir, "result-test.pkl")
    remote_stdout_filename = os.path.join(current_remote_workdir, "stdout.log")
    remote_stderr_filename = os.path.join(current_remote_workdir, "stderr.log")

    def mock_open(*args):
        mock_file_object = mock.MagicMock()

        if os.path.basename(args[0]) == os.path.basename(remote_result_filename):
            mock_file_object.__aenter__.return_value.read.return_value = None
        elif os.path.basename(args[0]) == os.path.basename(remote_stdout_filename):
            mock_file_object.__aenter__.return_value.read.return_value = expected_stdout
        elif os.path.basename(args[0]) == os.path.basename(remote_stderr_filename):
            mock_file_object.__aenter__.return_value.read.return_value = expected_stderr
        else:
            raise Exception(f"incorrect mocking.{[args[0]]}")

        return mock_file_object

    mocker.patch("aiofiles.open", side_effect=mock_open)

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=current_remote_workdir,
    )

    results, stdout, stderr, error = await executor.query_result(
        conn_mock,
        remote_result_filename,
        remote_stdout_filename,
        remote_stderr_filename,
        task_results_dir,
    )
    assert results == expected_results
    assert stdout == expected_stdout
    assert stderr == expected_stderr
    assert error == expected_error
    assert patch_remove.call_count == 3

    # Reset the call_count
    patch_remove.call_count = 0

    # Check that aiofiles.os.remove is only called twice when remote_stderr_filename is same as remote_stdout_filename.
    results, stdout, stderr, error = await executor.query_result(
        conn_mock,
        remote_result_filename,
        remote_stdout_filename,
        remote_stdout_filename,
        task_results_dir,
    )
    assert patch_remove.call_count == 2


@pytest.mark.asyncio
async def test_query_result_raise_error(conn_mock, proc_mock):
    """Test that query_result raises error if the results file cannot be found."""

    # Now mock result files.
    proc_mock.returncode = 1
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    current_remote_workdir = "/home/test_user/workdir"
    task_results_dir = "/path/to/results_dir"
    remote_result_filename = os.path.join(current_remote_workdir, "result-test.pkl")
    remote_stdout_filename = os.path.join(current_remote_workdir, "stdout.log")
    remote_stderr_filename = os.path.join(current_remote_workdir, "stderr.log")

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=current_remote_workdir,
    )

    with pytest.raises(FileNotFoundError):
        await executor.query_result(
            conn_mock,
            remote_result_filename,
            remote_stdout_filename,
            remote_stderr_filename,
            task_results_dir,
        )


@pytest.mark.asyncio
async def test_run(mocker, tmpdir, conn_mock, proc_mock):
    """Test calling run works as expected."""

    # dummy objects
    def f(x, y):
        return x + y

    dummy_function = partial(mock_wrapper_fn, TransportableObject(f))
    dummy_task_metadata = {
        "dispatch_id": "378cdd10-8443-11ee-99e2-a35cb4664dfd",
        "node_id": "1",
        "results_dir": "path/to/results_dir",
    }
    dummy_func_args = [1]
    dummy_func_kwargs = {"y": 2}
    dummy_result = f(*dummy_func_args, **dummy_func_kwargs)

    # To memorize local variables of run()
    local_variables_run = mock.Mock()

    # Mocking behaviors
    def mock_format_submit_script(
        executor, py_version_func, py_script_filename, current_remote_workdir
    ):
        local_variables_run.py_version_func = py_version_func
        local_variables_run.py_script_filename = py_script_filename
        local_variables_run.current_remote_workdir = current_remote_workdir

        return submit_script_str

    async def mock_upload_task(
        executor,
        conn,
        func_filename,
        py_script_filename,
        submit_script_filename,
        current_remote_workdir,
    ):
        local_variables_run.func_filename = func_filename
        local_variables_run.py_script_filename = py_script_filename
        local_variables_run.submit_script_filename = submit_script_filename

    job_id = "00000001"

    async def mock_submit_task(*_):
        proc_mock.stdout = f"Your job {job_id} (\"jobscript-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.sh\") has been submitted"
        return proc_mock

    async def mock_poll_task(executor, conn, job_id):
        local_variables_run.job_id = job_id
        return

    mock_stdout = "mock stdout of the job"
    mock_stderr = "mock stderr of the job"
    mock_exception = None

    async def mock_query_result(
        executor,
        conn,
        remote_result_filename,
        remote_stdout_filename,
        remote_stderr_filename,
        task_results_dir,
    ):
        local_variables_run.remote_result_filename = remote_result_filename
        local_variables_run.remote_stdout_filename = remote_stdout_filename
        local_variables_run.remote_stderr_filename = remote_stderr_filename
        local_variables_run.task_results_dir = task_results_dir

        return dummy_result, mock_stdout, mock_stderr, mock_exception

    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.__aenter__.return_value.run = mock.AsyncMock(return_value=proc_mock)
    mocker.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )

    py_script_str = "mock python script"
    mocker.patch.object(GridEngineExecutor, "_format_py_script", return_value=py_script_str)

    submit_script_str = "mock submit script"
    mocker.patch.object(GridEngineExecutor, "_format_submit_script", new=mock_format_submit_script)

    mocker.patch.object(GridEngineExecutor, "_upload_task", new=mock_upload_task)
    mocker.patch.object(GridEngineExecutor, "submit_task", new=mock_submit_task)
    mocker.patch.object(GridEngineExecutor, "set_job_handle", return_value=mock.AsyncMock())
    mocker.patch.object(GridEngineExecutor, "_poll_task", new=mock_poll_task)
    mocker.patch.object(GridEngineExecutor, "query_result", new=mock_query_result)
    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)

    # Use temporary directory as cache_dir
    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        create_unique_workdir=False,
        cache_dir=str(tmpdir),
    )
    executor._task_stdout = io.StringIO()
    executor._task_stderr = io.StringIO()

    func_filename = (
        f"func-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.pkl"
    )
    submit_script_filename = (
        f"jobscript-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.sh"
    )
    py_script_filename = (
        f"script-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.py"
    )
    result_filename = (
        f"result-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.pkl"
    )

    actual_result = await executor.run(
        dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
    )

    # Check the return value of run.
    assert actual_result == dummy_result

    # Check local variables of run.
    assert local_variables_run.task_results_dir == os.path.join(
        dummy_task_metadata["results_dir"], dummy_task_metadata["dispatch_id"]
    )
    assert local_variables_run.py_version_func == ".".join(
        dummy_function.args[0].python_version.split(".")[:2]
    )
    assert local_variables_run.current_remote_workdir == remote_workdir

    # Check extraction of job_id works as expected.
    assert local_variables_run.job_id == job_id

    # Check whether stdout and stderr obtained from query_result are written in _task_stdout and _task_stderr.
    assert executor._task_stdout.getvalue() == mock_stdout
    assert executor._task_stderr.getvalue() == mock_stderr

    # Check wheter private instance variables for teardown are set as expected.
    assert executor._remote_func_filename == os.path.join(remote_workdir, func_filename)
    assert executor._remote_job_script_filename == os.path.join(
        remote_workdir, submit_script_filename
    )
    assert executor._remote_py_script_filename == os.path.join(remote_workdir, py_script_filename)
    assert executor._remote_result_filename == os.path.join(remote_workdir, result_filename)
    assert executor._remote_stdout_filename == f"{submit_script_filename}.o{job_id}"
    assert executor._remote_stderr_filename == f"{submit_script_filename}.e{job_id}"

    # Check whether writing scripts works as expected.
    with open(local_variables_run.submit_script_filename) as job_script:
        assert job_script.read() == submit_script_str
    with open(local_variables_run.py_script_filename) as py_script:
        assert py_script.read() == py_script_str

    # Check whether current_workdir is the expected name if create_unique_workdir is True.
    executor_with_unique_workdir = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        create_unique_workdir=True,
        cache_dir=str(tmpdir),
    )
    executor_with_unique_workdir._task_stdout = io.StringIO()
    executor_with_unique_workdir._task_stderr = io.StringIO()

    await executor_with_unique_workdir.run(
        dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
    )

    assert local_variables_run.current_remote_workdir == os.path.join(
        remote_workdir,
        dummy_task_metadata["dispatch_id"],
        f"node_{dummy_task_metadata['node_id']}",
    )


@pytest.mark.asyncio
async def test_run_handling_error(mocker, tmpdir, conn_mock, proc_mock):
    """Test function run handling error as expected."""

    # Use temporary directory as cache_dir
    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        create_unique_workdir=False,
        cache_dir=str(tmpdir),
    )
    executor._task_stdout = io.StringIO()
    executor._task_stderr = io.StringIO()

    # dummy objects
    def f(x, y):
        return x + y

    dummy_function = partial(mock_wrapper_fn, TransportableObject(f))
    dummy_task_metadata = {
        "dispatch_id": "378cdd10-8443-11ee-99e2-a35cb4664dfd",
        "node_id": "1",
        "results_dir": "path/to/results_dir",
    }
    dummy_func_args = [1]
    dummy_func_kwargs = {"y": 2}

    # Mocking behaviors
    async def mock_submit_task_failed(*_):
        proc_mock.returncode = 1

        return proc_mock

    job_id = "00000001"

    async def mock_submit_task_succeeded(*_):
        proc_mock.returncode = 0
        proc_mock.stdout = f"Your job {job_id} (\"jobscript-{dummy_task_metadata['dispatch_id']}-{dummy_task_metadata['node_id']}.sh\") has been submitted"

        return proc_mock

    async def mock_query_result_with_exception(*_):
        return None, "", "stderr", RuntimeError("mock runtime error")

    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.__aenter__.return_value.run = mock.AsyncMock(return_value=proc_mock)

    mocker.patch.object(GridEngineExecutor, "get_cancel_requested", return_value=False)
    mocker.patch.object(GridEngineExecutor, "set_job_handle", return_value=mock.AsyncMock())
    mocker.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )

    py_script_str = "mock python script"
    mocker.patch.object(GridEngineExecutor, "_format_py_script", return_value=py_script_str)

    submit_script_str = "mock submit script"
    mocker.patch.object(
        GridEngineExecutor, "_format_submit_script", return_value=submit_script_str
    )

    mocker.patch.object(GridEngineExecutor, "_upload_task", return_value=mock.AsyncMock())

    patch_submit_task_failed = mock.patch.object(
        GridEngineExecutor, "submit_task", new=mock_submit_task_failed
    )
    patch_submit_task_succeeded = mock.patch.object(
        GridEngineExecutor, "submit_task", new=mock_submit_task_succeeded
    )

    patch_poll_task_runtime_error = mock.patch.object(
        GridEngineExecutor, "_poll_task", side_effect=RuntimeError("runtime error")
    )
    patch_poll_task_cancelled_error = mock.patch.object(
        GridEngineExecutor, "_poll_task", side_effect=TaskCancelledError
    )
    patch_poll_task_succeeded = mock.patch.object(
        GridEngineExecutor, "_poll_task", return_value=None
    )

    patch_query_result_with_exception = mock.patch.object(
        GridEngineExecutor, "query_result", new=mock_query_result_with_exception
    )

    # Check whether raises RuntimeError if failed to make the working directory on the remote machine.
    proc_mock.returncode = 1
    with pytest.raises(RuntimeError) as err:
        await executor.run(dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata)
    # reset proc_mock
    proc_mock.returncode = 0

    # Check whether raises error if failed to submit the task.
    with patch_submit_task_failed:
        with pytest.raises(RuntimeError) as err:
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )
    # reset proc_mock
    proc_mock.returncode = 0

    # Case that _poll_task raises RuntimeError
    with patch_submit_task_succeeded, patch_poll_task_runtime_error:
        with pytest.raises(RuntimeError) as err:
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )

    # Case that _poll_task raises TaskCancelledError
    with patch_submit_task_succeeded, patch_poll_task_cancelled_error:
        with pytest.raises(TaskCancelledError) as err:
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )

    # Case that query_result raises an error.
    with patch_submit_task_succeeded, patch_poll_task_succeeded, patch_query_result_with_exception:
        with pytest.raises(TaskRuntimeError):
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )


@pytest.mark.asyncio
async def test_run_handling_cancel_request(mocker, tmpdir, conn_mock, proc_mock):
    """Test function run handling the request that cancel the task as expected."""

    # Use temporary directory as cache_dir
    remote_workdir = "/home/test_user/workdir"
    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        remote_workdir=remote_workdir,
        create_unique_workdir=False,
        cache_dir=str(tmpdir),
    )
    executor._task_stdout = io.StringIO()
    executor._task_stderr = io.StringIO()

    # dummy objects
    def f(x, y):
        return x + y

    dummy_function = partial(mock_wrapper_fn, TransportableObject(f))
    dummy_task_metadata = {
        "dispatch_id": "378cdd10-8443-11ee-99e2-a35cb4664dfd",
        "node_id": "1",
        "results_dir": "path/to/results_dir",
    }
    dummy_func_args = [1]
    dummy_func_kwargs = {"y": 2}

    # Mocking behaviors
    proc_mock.returncode = 0
    proc_mock.stdout = "stdout"
    proc_mock.stderr = "stderr"
    conn_mock.__aenter__.return_value.run = mock.AsyncMock(return_value=proc_mock)

    patch_client_connect = mocker.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )

    py_script_str = "mock python script"
    patch_format_py_script = mocker.patch.object(
        GridEngineExecutor, "_format_py_script", return_value=py_script_str
    )

    submit_script_str = "mock submit script"
    patch_format_submit_script = mocker.patch.object(
        GridEngineExecutor, "_format_submit_script", return_value=submit_script_str
    )

    patch_upload_task = mocker.patch.object(
        GridEngineExecutor, "_upload_task", return_value=mock.AsyncMock()
    )

    def get_cancel_requested_return_value(times: int):
        ret = [False] * times
        ret[-1] = True
        return ret

    # Case request that task to be cancelled before calling _client_connect.
    with mock.patch.object(
        GridEngineExecutor,
        "get_cancel_requested",
        side_effect=get_cancel_requested_return_value(1),
    ):
        with pytest.raises(TaskCancelledError):
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )
        assert patch_client_connect.call_count == 0

    # Case request that task to be cancelled after calling _client_connect.
    with mock.patch.object(
        GridEngineExecutor,
        "get_cancel_requested",
        side_effect=get_cancel_requested_return_value(2),
    ):
        with pytest.raises(TaskCancelledError):
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )
        assert patch_client_connect.call_count == 1
        assert patch_format_py_script.call_count == 0

    # Case request that task to be cancelled before uploading task to remote machine.
    with mock.patch.object(
        GridEngineExecutor,
        "get_cancel_requested",
        side_effect=get_cancel_requested_return_value(3),
    ):
        with pytest.raises(TaskCancelledError):
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )
        assert patch_format_submit_script.call_count == 1
        assert patch_upload_task.call_count == 0

    # Case request that task to be cancelled before submitting task to Grid Engine.
    with mock.patch.object(
        GridEngineExecutor,
        "get_cancel_requested",
        side_effect=get_cancel_requested_return_value(4),
    ):
        with pytest.raises(TaskCancelledError):
            await executor.run(
                dummy_function, dummy_func_args, dummy_func_kwargs, dummy_task_metadata
            )
        assert patch_upload_task.call_count == 1


@pytest.mark.asyncio
async def test_perform_cleanup(mocker, conn_mock, proc_mock):
    """Test perform_cleanup works as expected."""

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    proc_mock.returncode = 0
    proc_mock.stdout = ""
    proc_mock.stderr = ""
    conn_mock.run = mock.AsyncMock(return_value=proc_mock)

    patch_async_os_remove = mocker.patch("aiofiles.os.remove", return_value=mock.AsyncMock())

    local_cache_dir = "/home/.cache/covalent/task1"
    executor._local_func_filename = os.path.join(local_cache_dir, "func-local-1.pkl")
    executor._local_job_script_filename = os.path.join(local_cache_dir, "jobscript-local-1.sh")
    executor._local_py_script_filename = os.path.join(local_cache_dir, "script-local-1.py")

    remote_workdir = "/home/test_user/workdir"
    executor._remote_func_filename = os.path.join(remote_workdir, "func-test-1.pkl")
    executor._remote_job_script_filename = os.path.join(remote_workdir, "job_script-test-1.sh")
    executor._remote_py_script_filename = os.path.join(remote_workdir, "py_script-test-1.py")
    executor._remote_result_filename = os.path.join(remote_workdir, "result-test-1.pkl")
    executor._remote_stdout_filename = os.path.join(remote_workdir, "stdout.log")
    executor._remote_stderr_filename = os.path.join(remote_workdir, "stderr.log")

    await executor._perform_cleanup(conn=conn_mock)
    assert patch_async_os_remove.call_count == 3
    assert conn_mock.run.call_count == 6


@pytest.mark.asyncio
async def test_teardown(mocker, conn_mock):
    """Test teardown works as expected."""

    executor_not_cleanup = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        cleanup=False,
    )

    conn_mock.wait_closed = mock.AsyncMock()

    mocker.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )

    patch_perform_cleanup = mocker.patch.object(
        GridEngineExecutor, "_perform_cleanup", new=mock.AsyncMock(return_value=None)
    )

    # Test _perform_cleanup is not called if cleanup is False.
    await executor_not_cleanup.teardown(task_metadata={})
    assert patch_perform_cleanup.call_count == 0

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        cleanup=True,
    )

    local_cache_dir = "/home/.cache/covalent/task1"
    remote_workdir = "/home/test_user/workdir"

    executor._local_func_filename = os.path.join(local_cache_dir, "func-local-1.pkl")
    executor._local_job_script_filename = os.path.join(local_cache_dir, "jobscript-local-1.sh")
    executor._local_py_script_filename = os.path.join(local_cache_dir, "script-local-1.py")
    executor._remote_func_filename = os.path.join(remote_workdir, "func-test-1.pkl")
    executor._remote_job_script_filename = os.path.join(remote_workdir, "job_script-test-1.sh")
    executor._remote_py_script_filename = os.path.join(remote_workdir, "py_script-test-1.py")
    executor._remote_result_filename = os.path.join(remote_workdir, "result-test-1.pkl")
    executor._remote_stdout_filename = os.path.join(remote_workdir, "stdout.log")
    executor._remote_stderr_filename = os.path.join(remote_workdir, "stderr.log")

    # Test _perform_cleanup is called if cleanup is True.
    await executor.teardown(task_metadata={})
    assert patch_perform_cleanup.call_count == 1


@pytest.mark.asyncio
async def test_teardown_handling_error(conn_mock):
    """Test teardown handles exceptions as expected."""

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
        cleanup=True,
    )

    local_cache_dir = "/home/.cache/covalent/task1"
    remote_workdir = "/home/test_user/workdir"

    executor._local_func_filename = os.path.join(local_cache_dir, "func-local-1.pkl")
    executor._local_job_script_filename = os.path.join(local_cache_dir, "jobscript-local-1.sh")
    executor._local_py_script_filename = os.path.join(local_cache_dir, "script-local-1.py")
    executor._remote_func_filename = os.path.join(remote_workdir, "func-test-1.pkl")
    executor._remote_job_script_filename = os.path.join(remote_workdir, "job_script-test-1.sh")
    executor._remote_py_script_filename = os.path.join(remote_workdir, "py_script-test-1.py")
    executor._remote_result_filename = os.path.join(remote_workdir, "result-test-1.pkl")
    executor._remote_stdout_filename = os.path.join(remote_workdir, "stdout.log")
    executor._remote_stderr_filename = os.path.join(remote_workdir, "stderr.log")

    conn_mock.wait_closed = mock.AsyncMock()

    patch_client_connect_succeeded = mock.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )
    patch_client_connect_failed = mock.patch.object(
        GridEngineExecutor, "_client_connect", side_effect=RuntimeError("failed _client_connect")
    )

    patch_perform_cleanup_failed = mock.patch.object(
        GridEngineExecutor, "_perform_cleanup", side_effect=Exception("Mock error")
    )

    # teardown does not raise an exception.
    with patch_client_connect_failed:
        await executor.teardown(task_metadata={})

    with patch_client_connect_succeeded, patch_perform_cleanup_failed:
        await executor.teardown(task_metadata={})


@pytest.mark.asyncio
async def test_cancel(mocker, conn_mock, proc_mock):
    """Test that cancel works as expected."""

    mocker.patch.object(
        GridEngineExecutor, "_client_connect", new=mock.AsyncMock(return_value=conn_mock)
    )

    async def mock_run_succeeded(cmd):
        proc_mock.returncode = 0
        proc_mock.cmd = cmd
        return proc_mock

    async def mock_run_failed(cmd):
        proc_mock.returncode = 1
        proc_mock.cmd = cmd
        return proc_mock

    conn_mock.wait_closed = mock.AsyncMock()

    job_id = "00000001"

    executor = GridEngineExecutor(
        username="test_user",
        address="test_address",
        ssh_key_file="~/.ssh/id_rsa",
    )

    # Check whether no error is raised in case that executing qdel is succeeded.
    # Also check whether executed command is correct.
    conn_mock.__aenter__.return_value.run = mock_run_succeeded
    await executor.cancel({}, job_id)
    assert proc_mock.cmd == f"qdel {job_id}"

    # Check whether no error is raised in case that executing qdel is failed.
    conn_mock.__aenter__.return_value.run = mock_run_failed
    await executor.cancel({}, job_id)

    # Check whether no error is raised in case that job_id is None.
    conn_mock.__aenter__.return_value.run = mock_run_failed
    await executor.cancel({}, None)


def test_make_remote_log_filenames():
    """Test that _make_remote_log_filenames works as expected."""

    job_id = "00000001"
    job_script_filename = "job_script-test-1.sh"

    embedded_jobname = "embedded_jobname"
    jobname = "jobname"
    embedded_test_dir = "embedded_test_dir/"
    test_dir = "test_dir/"

    # Without `-N`, `-o`, `-e`, and `-j y`, filenames are default.
    executor = GridEngineExecutor()
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == f"{job_script_filename}.e{job_id}"

    # If `-N` set in embedded_qsub_args, job_name will be the one specified in embedded_qsub_args.
    executor = GridEngineExecutor(embedded_qsub_args={"N": embedded_jobname})
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{embedded_jobname}.o{job_id}"
    assert remote_stderr_filename == f"{embedded_jobname}.e{job_id}"

    # If `-N` set in both qsub_args and embedded_qsub_args, that of qsub_args is preferred.
    executor = GridEngineExecutor(
        embedded_qsub_args={"N": embedded_jobname}, qsub_args={"N": jobname}
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{jobname}.o{job_id}"
    assert remote_stderr_filename == f"{jobname}.e{job_id}"

    # If a directory path is specified, the log file will be output with the default name under the directory.
    # If a file path is specified, log file will be output with the path.
    expected_stderr_filename = f"{embedded_test_dir}stderr.log"
    executor = GridEngineExecutor(
        embedded_qsub_args={"o": embedded_test_dir, "e": expected_stderr_filename}
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{embedded_test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == expected_stderr_filename

    # If `-o` or `-e` is set in both embedded_qsub_args and qsub_args, that of qsub_args is preferred.
    expected_stderr_filename = f"{test_dir}stderr.log"
    executor = GridEngineExecutor(
        embedded_qsub_args={"o": embedded_test_dir, "e": f"{embedded_test_dir}stderr.log"},
        qsub_args={"o": test_dir, "e": expected_stderr_filename},
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == expected_stderr_filename

    # If `-j y[es]` is set, remote_stderr_filename is same as remote_stdout_filename.
    executor = GridEngineExecutor(embedded_qsub_args={"j": "y"})
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == remote_stdout_filename

    executor = GridEngineExecutor(embedded_qsub_args={"j": "yes"})
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == remote_stdout_filename

    # If `-j n[o]` is set, remote_stderr_filename is different from remote_stdout_filename.
    expected_stderr_filename = f"{embedded_test_dir}stderr.log"
    executor = GridEngineExecutor(
        embedded_qsub_args={"o": embedded_test_dir, "e": expected_stderr_filename, "j": "n"}
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{embedded_test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == expected_stderr_filename

    executor = GridEngineExecutor(
        embedded_qsub_args={"o": embedded_test_dir, "e": expected_stderr_filename, "j": "no"}
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{embedded_test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == expected_stderr_filename

    # If `-j y[es]|n[o]` is set both in embedded_qsub_args and qsub_args, that of qsub_args is preferred.
    expected_stderr_filename = f"{test_dir}stderr.log"
    executor = GridEngineExecutor(
        embedded_qsub_args={
            "o": embedded_test_dir,
            "e": f"{embedded_test_dir}stderr.log",
            "j": "y",
        },
        qsub_args={
            "o": test_dir,
            "e": expected_stderr_filename,
            "j": "n",
        },
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == expected_stderr_filename

    executor = GridEngineExecutor(
        embedded_qsub_args={
            "o": embedded_test_dir,
            "e": f"{embedded_test_dir}stderr.log",
            "j": "n",
        },
        qsub_args={
            "o": test_dir,
            "e": expected_stderr_filename,
            "j": "y",
        },
    )
    remote_stdout_filename, remote_stderr_filename = executor._make_remote_log_filenames(
        job_id, job_script_filename
    )
    assert remote_stdout_filename == f"{test_dir}{job_script_filename}.o{job_id}"
    assert remote_stderr_filename == remote_stdout_filename
