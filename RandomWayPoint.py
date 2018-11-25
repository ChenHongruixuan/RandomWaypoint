import numpy as np
import matplotlib.pyplot as plt


class RandomWayPoint(object):
    def __init__(self, steps, x_range, y_range):
        np.random.seed()
        self.steps = steps
        self.pause_max = 5
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]

    def generate_trace(self, start_coor):
        """accord modified version of the RWP to generate trace data"""
        trace_data = []
        start_x = start_coor[0]
        start_y = start_coor[1]
        dest_x, dest_y = self._generate_dest_area()
        velo = self._generate_velo()
        step = self.steps
        while step > 0:
            part_trace, step_remain = self._generate_part_trace(start_x, start_y,
                                                                dest_x, dest_y, velo, step_limit=step)

            trace_data.extend(part_trace)
            if step_remain == 0:
                break
            step = step_remain
            pause_t = self._generate_pause(step)
            last_coor = trace_data[-1]
            for p in range(pause_t):
                trace_data.append(last_coor)
            step -= pause_t
            start_x = trace_data[-1][0]
            start_y = trace_data[-1][1]
            dest_x, dest_y = self._generate_dest_area()
            velo = self._generate_velo()
        return np.array(trace_data)

    def _generate_part_trace(self, start_x, start_y, dest_x, dest_y, velo, step_limit):
        """accord modified version of the RWP to generate the part trace data"""
        part_trace = []

        step_remain = step_limit
        next_x, next_y = start_x, start_y
        angle = np.arctan2(dest_y - start_y, dest_x - start_x)
        step_x = velo * np.cos(angle)
        step_y = velo * np.sin(angle)

        for step in range(step_limit, 0, -1):
            step_remain -= 1
            if np.abs(next_x - dest_x) < np.abs(step_x):
                next_x = dest_x
                next_y = dest_y
                part_trace.append(np.array([next_x, next_y]))
                break
            else:
                next_x += step_x
                next_y += step_y
                part_trace.append(np.array([next_x, next_y]))
        return part_trace, step_remain

    def _generate_dest_area(self):
        """generate a random dest area locating in limited area"""
        pro = np.random.rand()
        if pro > 0.35:
            dest_x = np.random.uniform(self.x_min, self.x_max)
            dest_y = np.random.uniform(self.y_min, self.y_max)
        elif 0.15 < pro <= 0.35:
            dest_x = int(np.random.uniform(self.x_min, self.x_max)) + 0.01
            dest_y = int(np.random.uniform(self.y_min, self.y_max)) + 0.01
        else:
            type = np.random.randint(0, 4)
            if type == 0:
                dest_x = np.random.uniform(self.x_min, self.x_max)
                dest_y = self.y_min + 0.01
            elif type == 1:
                dest_x = self.x_max - 0.01
                dest_y = np.random.uniform(self.y_min, self.y_max)
            elif type == 2:
                dest_x = np.random.uniform(self.x_min, self.x_max)
                dest_y = self.y_max - 0.01
            else:
                dest_x = self.x_min + 0.01
                dest_y = np.random.uniform(self.y_min, self.y_max)
        return dest_x, dest_y

    def _generate_pause(self, p_max):
        """generate a pause time"""
        pause = np.random.randint(0, min(p_max, self.pause_max))
        return pause

    def _generate_velo(self):
        """generate random velocity between v_min and v_max"""
        velo = np.random.uniform(0.3, 1.3)  # every step  cannot exceed a certain number
        return velo

    def _get_epsilon(self):
        return np.random.uniform(-0.1, 0.1)


def test_model():
    x_range = np.array([0, 11])
    y_range = np.array([0, 11])
    model = RandomWayPoint(steps=500, x_range=x_range, y_range=y_range)
    trace_data = model.generate_trace(start_coor=[1, 1])
    # draw generated trace data
    plt.ion()
    plt.show()
    plt.figure(0)
    plt.plot(trace_data[:, 0], trace_data[:, 1], 'r')
    plt.draw()
    plt.pause(10)


if __name__ == '__main__':
    test_model()
