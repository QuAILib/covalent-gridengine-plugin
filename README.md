# README

## Covalent Grid Engine Plugin

Covalent is a Pythonic workflow tool used to execute tasks on advanced computing hardware. This executor plugin interfaces Covalent with HPC systems managed by [Grid Engine](https://altair.com/grid-engine). For workflows to be deployable, users must have SSH access to the Grid Engine login node, writable storage space on the remote filesystem, and permissions to submit jobs to Grid Engine.

## Installation

### To install from PyPI

To use this plugin with Covalent, simply install it using `pip` in whatever Python environment you use to run the Covalent server (your local machine by default):

```console
pip install covalent-gridengine-plugin
```

### To install from the source

If you install this plugin directly from the source, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory:

```shell
cd this-repository
```

3. Install this plugin using pip:

```shell
pip install .
```

### Environment on the remote system

On the remote system, the Python major and minor version numbers on both the local and remote machines must match to ensure reliable (un)pickling of the various objects. Additionally, the remote system's Python environment must have the base [covalent package](https://github.com/AgnostiqHQ/covalent) installed (e.g. `pip install covalent`).

## Usage

### Using the Plugin in a Workflow: Approach 1

With your [Covalent config file](https://docs.covalent.xyz/docs/user-documentation/how-to/customization/)(found at `~/.config/covalent/covalent.conf` by default) appropriately set up, one can run a workflow on the HPC machine as follows:

```python
import covalent as ct

@ct.electron(executor="gridengine")
def add(a, b):
    return a + b

@ct.lattice
def workflow(a, b):
    return add(a, b)


dispatch_id = ct.dispatch(workflow)(1, 2)
result = ct.get_result(dispatch_id)

```

### Using the Plugin in a Workflow: Approach 2

If you wish to modify the various parameters within your Python script rather than solely relying on the Covalent configuration file, it is possible to do that as well by instantiating a custom instance of the GridEngineExecutor class. An example with some commonly used parameters is shown below. By default, any parameters not specified in the GridEngineExecutor will be inherited from the configuration file.

```python
import covalent as ct

executor = ct.executor.GridEngineExecutor(
    username="UserName",
    address="remotemachine.address",
    port=22,
    ssh_key_file="~/.ssh/id_rsa",
    remote_workdir="$HOME/remote_workdir",
    poll_freq=30,
    cleanup=True,
    embedded_qsub_args={
        "l": ["rt_F=1", "h_rt=1:00:00"],
    },
    qsub_args={
        "ar": "ar_id",
    },
    prerun_commands=[
        "source /etc/profile.d/modules.sh",
        "module load python/3.10/3.10.10",
        "module load cuda/11.8/11.8.0",
        "module load cudnn/8.8/8.8.1",
        "source ~/remote_workdir/.venv/bin/activate",
    ],
    bashrc_path="~/.bashrc",
)

@ct.electron(executor=executor)
def add(a, b):
    return a + b

@ct.lattice
def workflow(a, b):
    return add(a, b)


dispatch_id = ct.dispatch(workflow)(1, 2)
result = ct.get_result(dispatch_id)
```

### Configuration

There are many configuration options that can be passed in to the class `ct.executor.GridEngineExecutor` or by modifying the covalent configuration file under the section `[executors.gridengine]`.

The following shows an example of how a user might modify their covalent configuration file to support this plugin:

```console
[executors.gridengine]
username = "UserName"
address = "remote_machine.example.address"
port = 22
ssh_key_file = "~/.ssh/id_rsa"
bashrc_path = "$HOME/.bashrc"
prerun_commands = ["module load ABC", "source ~/remote-workdir/.venv/bin/activate"]
cleanup = true
remote_workdir = "covalent-workdir"
create_unique_workdir = false
cache_dir = "~/.cache/covalent"
poll_freq = 30
remote_cache = ".cache/covalent"
log_stdout = "stdout.log"
log_stderr = "stderr.log"

[executors.gridengine.qsub_args]
P = "projectname"
ar = "ar_id"

[executors.gridengine.embedded_qsub_args]
l = ["h_rt=1:00:00", "arch=lx-amd64"]
soft = ""

```

#### Specifying Optional Paramters of `qsub` command

This plugin submits jobs to Grid Engine with `qsub` command. To specify the options for `qsub`, there are parameters `gridengine.qsub_args` and `embedded_qsub_args`, which specify default paramteres passed directly to `qsub` command and default paramtetes for `#$` directives in the job script, respectively.

For example, if the following is written in the configuration file,

```console
[executors.gridengine.qsub_args]
P = "projectname"
ar = "ar_id"

[executors.gridengine.embedded_qsub_args]
soft= ""
l = ["h_rt=1:00:00", "arch=lx-amd64"]

```

the task will be submitted by executing following command.

```
qsub -P projectname -ar ar_id {script_filename}
```

Also, the script submitted to Grid Engine contains the following directives.

```shell
#!/bin/bash
#$ -soft
#$ -l h_rt=1:00:00
#$ -l arch=lx-amd64
```

#### Other parameters

You can modify various parameters in the Covalent configuration file to better suit your needs, such as the address of the remote machine, the username to use when logging in, the ssh_key_file to use for authentication.

A full description of the various input parameters are described in the docstrings of the GridEngineExecutor class, reproduced below:

```python
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
```

## Release Notes

Release notes are available in the [Changelog](/CHANGELOG.md).

## Citation

Please use the following citation in any publications:

> W. J. Cunningham, S. K. Radha, F. Hasan, J. Kanem, S. W. Neagle, and S. Sanand.
> *Covalent.* Zenodo, 2022. https://doi.org/10.5281/zenodo.5903364

## License

Covalent is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/AgnostiqHQ/covalent-executor-template/blob/main/LICENSE) file or contact the [support team](mailto:support@agnostiq.ai) for more details.

## Acknowledgement

This work is based on results obtained from “Development of Quantum-Classical Hybrid Use-Case Technologies in Cyber-Physical Space” (JPNP23003), commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
