"""
噪声模型
集成了基础噪声模型和动力学特定噪声
"""
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import sin

class BaseNoiseModel():
    """
    base noise model
    """
    def __init__(self, frame: int) -> None:
        """
        Init: 
            frame: int : totol steps
            dt: single step time
        """
        # noise name
        self.name = 'base_noise'
        # frame lenght
        self.frame = frame
        # noise
        self.noise = np.zeros((frame, 6))

    def __call__(self) -> np.ndarray:

        return self.noise

    def visualize(self):
        fig = make_subplots(rows=3, cols=2)

        x = np.linspace(0, self.frame, self.frame+1)

        col_name = ['x', 'y', 'z', 'vx', 'vy', 'vz']

        for i in range(6):


            fig.add_trace(go.Scatter(x=x, y=self.noise[:,i],
                                    mode='lines+markers',
                                    name=col_name[i]),
                                    row=i%3 + 1,
                                    col=i//3 + 1 )
        fig.show()


class GaussianNoise(BaseNoiseModel):
    def __init__(self, frame: int, mean: np.ndarray, sigma:np.ndarray) -> None:
        super().__init__(frame)
        
        self.name = 'GaussianNoise'
        self.noise = np.zeros((frame, 6))

        for i in range(frame): 
            self.noise[i,:] = np.random.normal(mean, sigma, (1, 6))


class FlickerNoise(BaseNoiseModel):
    def __init__(self, frame: int, mean: np.ndarray, sigma_1:np.ndarray, sigma_2:np.ndarray, flicker_prob: float) -> None:
        super().__init__(frame)
        
        self.name = 'FlickerNoise'
        self.noise = np.zeros((frame, 6))

        for i in range(frame): 
            flag = np.random.uniform(0, 1)
            if flag > flicker_prob:
                self.noise[i,:] = np.random.normal(mean, sigma_1, (1, 6))
            else:
                self.noise[i,:] = np.random.normal(mean, sigma_2, (1, 6))

class TimeVaryNoise(BaseNoiseModel):
    def __init__(self, frame: int, mean: np.ndarray, sigma:np.ndarray, T: int) -> None:
        super().__init__(frame)
        
        self.name = 'TimeVaryNoise'
        self.noise = np.zeros((frame, 6))

        for i in range(frame): 

            self.noise[i,:] = np.random.normal(mean, sigma*(1-sin(i/T)), (1, 6))


class DriftNoise(BaseNoiseModel):
    def __init__(self, frame: int, mean: np.ndarray, sigma:np.ndarray, base:np.ndarray, dx: float) -> None:
        super().__init__(frame)
        
        self.name = 'TimeVaryNoise'
        self.noise = np.zeros((frame, 6))

        for i in range(frame): 
            scale =  dx * i * base
            self.noise[i,:] = np.random.normal(mean + scale, sigma, (1, 6))
