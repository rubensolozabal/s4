import datetime
from itertools import product
import sys, time
print(sys.executable)

from simple_slurm import Slurm

# experiment='lra/s4-listops'
# experiment='lra/s4-imdb'
# experiment='lra/s4-cifar'
experiment='lra/s4-aan'
# experiment='lra/s4-pathfinder'
# experiment='lra/s4-pathx'


# Create slurm object
slurm = Slurm(
        account='cscc-users',
        cpus_per_task=8,
        mem='128G',
        # array=range(0, 90),
        gres='gpu:1',
        # partition='it-hpc',
        qos='cscc-gpu-qos',
        partition='cscc-gpu-p',
        job_name=f'{experiment[-3:]}',
        output=f'slurm_logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        # time=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0),
        # exclude=['g512-1', 'g512-1']
    )

slurm.sbatch(f"python -m train experiment={experiment}")