from control_pendulum import control_pendulum

def test_basic():
  state,cost = control_pendulum(0, .03, 1/10)
