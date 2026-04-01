import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/GTL/aghitu/CS8813/tdmpc/turtlebot_rl/install/turtlebot_rl'
