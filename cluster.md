Connect to cluster via cli: `ssh [nethz]@student-cluster.inf.ethz.ch` (Password: your ethz mail password)

Copying files: `scp -r sourcedir [nethz]@student-cluster.inf.ethz.ch:targetdir` (Use absolute file paths, when copying again delete repo first)

Connect VSCode to cluster: Use remote-ssh extension, enter `[nethz]@student-cluster.inf.ethz.ch` and the same password before to login

After logging in the cluster:
- setting up cuda and pytorch/python packages [link](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda): 
    ```bash
    # add cuda
    module add cuda 12.9
    # create venv in home/[nethz]
    python3 -m venv ./venv 
    # activate venv
    source venv/bin/activate
    # install packages
    pip install --no-cache-dir -r ~/DL_project/requirements.txt
    python3 -m pip cache purge
    ```
- running jobs
  ```bash
  cd DL_project
  srun --pty -A deep_learning -t 120 python SinkhornTransport.py
  ```