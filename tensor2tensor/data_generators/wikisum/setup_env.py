import subprocess
import os

def main():
    cmd = ["export PATH=/opt/conda/bin:$PATH",
            "sudo apt-get install -y python3-setuptools",
            "pip install tensor2tensor --user"]
    if not os.path.exists('temp/tensor2tensor'):
        cmd.append('mkdir -p temp')
        cmd.append('cd temp')
        cmd.append('git clone https://github.com/armancohan/tensor2tensor.git')
    else:
        cmd.append('cd temp')
    cmd.append('cd tensor2tensor')
    cmd.append('pip3 install tensorflow-addons --user') 
    cmd.append('pip3 install -e . --user')
    cmd.append('cd ~')
    ssh_command = ';'.join(cmd)
    # ssh_command = '"bash -c ' + "'" + ';'.join(cmd) + "'" + '"'
    process = subprocess.Popen(
        ssh_command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    print("-" * 10)
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))    

if __name__ == '__main__':
    main()