apt update
apt install -y xvfb
Xvfb :0 -screen 0 2560x1440x24 -listen tcp -ac +extension GLX +extension RENDER &
export DISPLAY=:0

