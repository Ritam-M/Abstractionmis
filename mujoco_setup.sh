# We must install mujoco dependencies locally since we don't have root privileges.
# This script assumes you have already have all the nececesarry rpm packages in the ./rpm/ directory.
# You can get these as follows:
# (1) run `./mujoco_get_dependencies.sh` in your submit node.
# (2) moving the resulting `rpm` directory inside the directory containing all your code, 
# so that it is copied over with the rest of your code.

# HOME environment variable is set to the submit node's HOME, not the job's HOME,
# so we fetch the home path here.
WORKING_DIR="$(pwd)"

# Install mujoco dependencies
cd rpm
for rpm in `ls *.rpm`; do rpm2cpio $rpm | cpio -id ; done
ln -sf $WORKING_DIR/rpm/usr/lib64/libGL.so.1.7.0 $WORKING_DIR/rpm/usr/lib64/libGL.so

export PATH="$PATH:$WORKING_DIR/rpm/usr/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WORKING_DIR/rpm/usr/lib:$WORKING_DIR/rpm/usr/lib64"
export LDFLAGS="-L$WORKING_DIR/rpm/usr/lib -L$WORKING_DIR/rpm/usr/lib64"
export CPATH="$CPATH:$WORKING_DIR/rpm/usr/include"

cd ..

# Install mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WORKING_DIR/.mujoco/mujoco210/bin"

# Install mujoco-py
git clone https://github.com/openai/mujoco-py.git
pip install -e mujoco-py
export MUJOCO_PY_MUJOCO_PATH="$WORKING_DIR/.mujoco/mujoco210"
