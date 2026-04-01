#!/bin/bash
cd ~/CS8813/tdmpc/turtlebot_rl
colcon build
sed -i '1s|.*|#!/home/GTL/aghitu/.conda/envs/tdmpc2_312/bin/python|' ~/CS8813/tdmpc/turtlebot_rl/install/turtlebot_rl/lib/turtlebot_rl/goto
sed -i '1s|.*|#!/home/GTL/aghitu/.conda/envs/tdmpc2_312/bin/python|' ~/CS8813/tdmpc/turtlebot_rl/install/turtlebot_rl/lib/turtlebot_rl/experiment_runner
source install/setup.bash