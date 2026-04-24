import numpy as np

class ParameterTracker:
    def __init__(self):
        self.history = {
            "iteration": [],
            "loss": [],
            'grad_norm': []
        }
        self.parameter_history = {}

    def record_iteration(self, iteration, loss_value, model):
        """records the parameters of each iteration.
            :params
                iteration: current iteration number
                loss_value: loss value from the current iteration (scalar)
                model : MLP model instance for the current iteration
        """
        self.history['iteration'].append(iteration)
        if isinstance(loss_value, (int, float)):
            scalar_loss = loss_value
        else:
            scalar_loss = float(loss_value.data) if hasattr(loss_value, "data") else float(loss_value)
        self.history['loss'].append(scalar_loss)

        for idx, param in enumerate(model.parameters()):
            param_name = f"param_{idx}"
            if param_name not in self.parameter_history:
                self.parameter_history[param_name] = []
            param_value = np.copy(param.data)
            self.parameter_history[param_name].append(param_value)

    def record_gradients(self, model):
        """
        store gradient  norm for checking exploding or vanisihing gradients.
        """
        total_norm = 0.0
        for param in model.parameters():
            squared_grad_sum = np.sum(param.grad) ** 2
            total_norm += squared_grad_sum
            
        self.history['grad_norm'].append(total_norm)


def zero_grad(model):
    """set all gradients to zero before backward pass"""
    for param in model.parameters():
        param.grad = np.zeros_like(param.data)