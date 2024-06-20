import torch

class PSPS2(torch.optim.Optimizer):

    def __init__(
            self,
            params,
            eps=1e-8):

        defaults = dict(eps=eps)

        super().__init__(params, defaults)

        self._update_precond_grad = self._update_precond_grad_cg

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["v"] = torch.flatten(torch.zeros_like(p))

        self._step_t = 0


    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_t += 1
        self._update_precond_grad()
        self.update(loss=loss)


        return loss

    def update(self, loss):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                precond_grad = state["precond_grad"]
                flat_grad = torch.flatten(p.grad.detach().clone())
                grad_norm_sq = torch.dot(flat_grad, precond_grad)
                eps = group['eps']
                if 2 * loss <= grad_norm_sq:
                    det = 1 - ( (2 * loss) / (grad_norm_sq) )
                    group["step_size"] = 1 - torch.sqrt(det).item()
                else:
                    group["step_size"] = 1.0

                # group["step_size"] = 0.01 # If you do this then this optimizer works for training NN too
                with torch.no_grad():
                    p.sub_(precond_grad.view_as(p), alpha=group['step_size'])

    def _update_precond_grad_identity(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["precond_grad"] = torch.flatten(p.grad)


    def hvp_from_grad(self, grads_tuple, list_params, vec_tuple):
        # don't damage grads_tuple. Grads_tuple should be calculated with create_graph=True
        dot = 0.
        for grad, vec in zip(grads_tuple, vec_tuple):
            dot += grad.mul(vec).sum()
        return torch.autograd.grad(dot, list_params, retain_graph=True)[0]



    def _update_precond_grad_cg(self):

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p_flat = torch.flatten(p.detach().clone())
                grad_flat = torch.flatten(p.grad.detach().clone())

                s = torch.zeros_like(p_flat) # s = H_inv * grad
                r = grad_flat.clone()
                b = r.clone()
                MAX_ITER = p.shape[0] * 2

                for cg_step in range(MAX_ITER):
                    hvp = torch.flatten(torch.autograd.grad(p.grad, p, grad_outputs=b.view_as(p), retain_graph=True)[0])
                    # hvp = torch.autograd.grad(p.grad, p, grad_outputs=b, retain_graph=True)[0]
                    # hvp = torch.flatten(self.hvp_from_grad(list(p.grad), p, list(b.view_as(p))))

                    alpha_k = torch.dot(r, r) / torch.dot(b, hvp)
                    s = s + alpha_k * b
                    r_prev = r.clone()
                    r = r - alpha_k * hvp
                    if torch.norm(r) < 1e-4:
                        # Ax = torch.autograd.grad(g, w, grad_outputs=s, retain_graph=True)[0]
                        # diff = torch.norm(Ax - f_grad)
                        break

                    beta_k = torch.dot(r, r) / torch.dot(r_prev, r_prev)
                    b = r + beta_k * b

                state["precond_grad"] = s


    def _update_precond_grad_adagrad(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                flat_grad = torch.flatten(p.grad.detach().clone())
                state["v"] = state["v"] + torch.square(flat_grad)
                precond = 1 / (torch.sqrt( state["v"]) + 1e-10)
                state["precond_grad"] = torch.mul(precond, flat_grad)

