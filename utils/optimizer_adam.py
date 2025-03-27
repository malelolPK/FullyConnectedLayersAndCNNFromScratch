import torch

# 0.003   0.000095
# 1e-4 5e-4
# 0.00001
class Adam_Optimizer:
    def __init__(self, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_dw = None
        self.v_dw = None
        self.m_db = None
        self.v_db = None

    def update(self, w, dw, b, db, t):
        t +=1
        # Initialize moments and velocities on first call
        if self.m_dw is None:
            self.m_dw = torch.zeros_like(w)
            self.v_dw = torch.zeros_like(w)
            self.m_db = torch.zeros_like(b)
            self.v_db = torch.zeros_like(b)

        # Update the first moment (m) and second moment (v)
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        # Bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)

        # Update weights and biases separately
        w_new = w - self.eta * m_dw_corr / (torch.sqrt(v_dw_corr) + self.epsilon)
        b_new = b - self.eta * m_db_corr / (torch.sqrt(v_db_corr) + self.epsilon)

        return w_new, b_new