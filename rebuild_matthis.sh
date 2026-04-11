#!/bin/bash
source /cs-share/pradalier/venv/conda.bash
conda activate tdmpc2_test
cd ~/ros2_ws/src/turtlebot_rl/
colcon build
sed -i '1s|.*|#!/home/GTL/mbernard/.conda/envs/tdmpc2_test/bin/python|' ~/ros2_ws/src/turtlebot_rl/install/turtlebot_rl/lib/turtlebot_rl/goto_odom
sed -i '1s|.*|#!/home/GTL/mbernard/.conda/envs/tdmpc2_test/bin/python|' ~/ros2_ws/src/turtlebot_rl/install/turtlebot_rl/lib/turtlebot_rl/experiment_runner
source install/setup.bash