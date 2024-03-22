import subprocess

subprocess.run("python run.py --ks 3 --rank 1 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 3 --rank 2 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 3 --rank 3 --num_seeds 50", shell=True)

subprocess.run("python run.py --ks 5 --rank 1 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 5 --rank 2 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 5 --rank 3 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 5 --rank 4 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 5 --rank 5 --num_seeds 50", shell=True)

subprocess.run("python run.py --ks 7 --rank 1 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 2 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 3 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 4 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 5 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 6 --num_seeds 50", shell=True)
subprocess.run("python run.py --ks 7 --rank 7 --num_seeds 50", shell=True)
