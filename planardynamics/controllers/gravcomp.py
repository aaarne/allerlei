class GravityCompensation:
    def __init__(self, robot):
        self._robot = robot

    def __call__(self, t, q, dq):
        return self._robot.G(q)
