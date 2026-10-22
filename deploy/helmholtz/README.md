# User-service deployment bundle

`prepare.py` plans or writes unit files for a dedicated Linux user service.
It never installs, enables, starts, stops, or restarts services. This deployment
recipe is separate from the general Docker/Compose stack.

Create a configuration with absolute paths to the checkout, complex-PETSc
Python, Redis executable, and a new service-state directory:

```json
{
  "repo": "/absolute/path/to/dtcc-sim",
  "python": "/absolute/path/to/complex-environment/bin/python",
  "redis": "/absolute/path/to/service-tools/bin/redis-server",
  "state_dir": "/absolute/path/to/helmholtz-service-state",
  "broker_port": 16379,
  "api_port": 18081,
  "worker_cpus": 16,
  "worker_memory": "64G",
  "job_timeout": 3600
}
```

Plan first, then write a new bundle for review. Paths containing `%`, `$`, or
newlines are refused because systemd interprets them specially.

```bash
python deploy/helmholtz/prepare.py --config /path/to/config.json --output /path/to/bundle
python deploy/helmholtz/prepare.py --config /path/to/config.json --output /path/to/bundle --write
systemd-analyze --user verify /path/to/bundle/*.service /path/to/bundle/*.target
```

The bundle starts Redis and the API only on `127.0.0.1`. It advertises and
registers only `helmholtz_simulation` using `DTCC_DATASET_ALLOWLIST`. Leaving
that setting empty preserves the general service's full dataset registry.
The allowlist controls availability; it is not authentication or a filesystem
sandbox. The dedicated API has no worker for other simulation families.

Redis uses AOF persistence with an every-second fsync and a no-eviction policy;
this can still lose recent writes on a crash. Broker restart/durability and
resource-limit behavior need deployment acceptance testing. Redis is capped at
256 MB internally and 512 MB for its service; the API is capped at 2 GB. Worker
CPU/memory limits are configurable; defaults are review proposals, not validated
budgets for every mesh. The worker also sets `CPUAffinity` to a selected CPU set,
inherited by its jobs and patch workers. By default the renderer selects the
first `worker_cpus` IDs from its allowed Linux CPU set; render on the target
host. Optional `worker_cpu_ids` supplies an explicit unique allowed list of that
length. Rendering fails if it cannot select enough CPUs.

A cgroup limit covers the job and patch descendants even
though each job has its own process session. Limits require suitable controller
delegation to the user manager. In particular, a configured `CPUQuota` does not
prove enforcement: check `cpu.max` in the worker cgroup. A host can delegate
memory and pids without delegating cpu; inherited CPU affinity still bounds the
worker's available CPUs in that case. It is a CPU-set limit, not a scheduler-time
quota. Oversized jobs can be killed by the memory limit. Capacity and performance
for large meshes remain separate from checking the kernel settings.

Services restart on failure with a bounded start rate. Worker stop sends SIGTERM
to the main worker, allowing its current job to finish; after the job timeout
plus 15 seconds, systemd can force-stop the whole cgroup. Cancel jobs through
the API before planned maintenance when waiting is undesirable. A forced cgroup
stop can kill the watchdog too: check retained job records and task state after
such stops. Restarting the worker does not resume partial numerical cases or
guarantee exactly-once processing. Use a fresh case for a retry.

`service.env` is referenced by its rendered absolute path; keep the bundle at a
stable location. It is written with mode 0600. State files, job logs, result
packages and Redis AOF data remain under `state_dir`; no automatic deletion is
configured. Set retention and backup policy before sustained use.

## Activation is a separate operation

Review available ports, executable paths, resource budgets and file permissions.
Verify runtime readiness in the selected environment. A user manager must be
available, and `loginctl show-user "$USER" -p Linger` must be considered for
running after logout. If `Linger=no`, arrange lingering with the server
administrator before relying on persistence across the last logout. The bundle
does not change that setting or require sudo to render.

Only after deployment approval, copy the four `.service`/`.target` files to
`~/.config/systemd/user/`, retain the environment file at its rendered path,
run `systemctl --user daemon-reload`, and enable/start `dtcc-helmholtz.target`.
Use `systemctl --user status dtcc-helmholtz-{broker,api,worker}.service` and
`journalctl --user -u dtcc-helmholtz-worker.service` for inspection. Stop the
target with `systemctl --user stop dtcc-helmholtz.target`; disabling it also
requires `systemctl --user disable dtcc-helmholtz.target`.

Do not expose this API or Redis publicly as part of this recipe. It provides
no public authentication, per-user path enforcement, or HTTP result download
endpoint. For an approved private deployment, an SSH tunnel can reach the
localhost API; result packages are available in the shared-results directory.

## Recorded acceptance scope

Temporary uniquely named runtime units verified automatic worker/API restarts,
Redis AOF recovery of queued work and cancellation markers after a broker
crash beyond the fsync interval, and persistence of completed task metadata
after a graceful broker restart. The recovered tiny two-worker job completed
with package fields matching its saved case. Runtime links and units were
removed afterward; no login enablement was performed.

Memory and pids limits were read back from the kernel. The test host delegated
only memory and pids controllers, so the cgroup CPU quota was unavailable.
A two-CPU affinity mask was enforced on the worker and inherited by its job.
The 16-CPU normal proposal is separate from that bounded test. No memory-limit
stress test or large-mesh performance validation was run. Check your target
host's controller delegation and allowed CPU IDs when deploying elsewhere.
