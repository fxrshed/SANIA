def rademacher_old(weights):
    return torch.round(torch.rand_like(weights)) * 2 - 1

def diag_estimate_old(weights, grad, iters):
    Ds = []
    for j in range(iters):
        z = rademacher_old(weights)
        with torch.no_grad():
            hvp = torch.autograd.grad(grad, weights, grad_outputs=z, retain_graph=True)[0]
        Ds.append((hvp*z))
    return torch.mean(torch.stack(Ds), 0)

def ttv(tuple_in):
    return torch.cat([t.view(-1) for t in tuple_in])

def flat_hessian(flat_grads, params):
    full_hessian = []
    for i in range(flat_grads.size()[0]):
        temp_hess = torch.autograd.grad(flat_grads[i], params, retain_graph=True)
        full_hessian.append(ttv(temp_hess))
    return torch.stack(full_hessian)

def run_psps2(dataset, epochs, precond_method, pcg_method="none", seed=0, **kwargs):

    torch.manual_seed(seed)
    np.random.seed(seed)

    data, target, dataloader = dataset

    eps = kwargs.get("eps", 1e-6)

    # torch.manual_seed(seed)
    
    # parameters
    w = torch.zeros(data.shape[1], device=device).requires_grad_()

    # save loss and grad size to history
    hist = []

    opt = Adam([w], lr=0.1)
    

    opt.zero_grad()
    loss = loss_function(w, data.to(device), target.to(device))
    g, = torch.autograd.grad(loss, w, create_graph=True)
    f_grad = g.clone().detach() 


    if precond_method == "none":
        D = torch.ones_like(w)
    elif precond_method == "hutch":
        alpha=0.1
        beta=0.999
        init_iters = kwargs.get("hutch_init_iters", 20_000)
        Dk = diag_estimate_old(w, g, init_iters)
    elif precond_method == "pcg":
        MAX_ITER = train_data.shape[1] * 2

    elif precond_method == "scaling_vec":
        scaling_vec = kwargs["scaling_vec"]
        D = (1 / scaling_vec)**2
    elif precond_method == "adam" or precond_method == "adam_m":
        D = torch.zeros_like(g)
        v = torch.zeros_like(g)
        step_t = torch.tensor(0.)
        betas = (0.9, 0.999)
    elif precond_method == "adagrad" or precond_method == "adagrad_m":
        D = torch.zeros_like(g)
        v = torch.zeros_like(g)

    if pcg_method == "hutch":
        alpha=0.1
        beta=0.999
        init_iters = 20_000
        Dk_pcg = diag_estimate_old(w, g, init_iters)
    elif pcg_method == "adam" or pcg_method == "adam_m":
        D_pcg = torch.zeros_like(g)
        v_pcg = torch.zeros_like(g)
        step_t_pcg = torch.tensor(0.)
        betas = (0.9, 0.999)
    elif pcg_method == "adagrad" or pcg_method == "adagrad_m":
        D_pcg = torch.zeros_like(g)
        v_pcg = torch.zeros_like(g)
    elif pcg_method == "none":
        D_pcg = torch.ones_like(g)
    elif pcg_method == "hess_diag":
        closure = lambda w: loss_function(w, data, target)
        hess = torch.autograd.functional.hessian(closure, w)
        # hess_diag_inv = 1 / torch.diag(torch.autograd.functional.hessian(closure, w))
        # hess_diag_inv = 1 / torch.diag(loss_hessian(w, data, target))
        # hess = flat_hessian(g,[w])
        D_pcg = 1 / torch.diag(hess)
    elif pcg_method == "hess":
        # closure = lambda w: loss_function(w, data, target)
        # hess_inv = torch.inverse(torch.autograd.functional.hessian(closure, w))
        hess_inv = torch.inverse(loss_hessian(w, data, target))
        D_pcg = hess_inv.clone()


    for epoch in range(epochs):
        
        # EVAL
        opt.zero_grad()
        loss = loss_function(w, data.to(device), target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        grad_norm_sq = torch.linalg.norm(g) ** 2  
        acc = (np.sign(data @ w.detach().numpy()) == target).sum() / target.shape[0]
        print(f"[{epoch}/{epochs}] | Loss: {loss.item()} | GradNorm^2: {grad_norm_sq.item()} | Accuracy: {acc}")
        hist.append([loss.item(), grad_norm_sq.item(), acc])
        # END EVAL
           
        for i, (batch_data, batch_target) in enumerate(dataloader): 
            
            # opt.zero_grad()
            # loss = loss_function(w, data.to(device), target.to(device))
            # g, = torch.autograd.grad(loss, w, create_graph=True)
            # grad_norm_sq = torch.linalg.norm(g) ** 2  
            # print(f"[{i}/{epoch}] | Loss: {loss.item()} | GradNorm^2: {grad_norm_sq.item()}")

            opt.zero_grad()
            loss = loss_function(w, batch_data, batch_target)
            g, = torch.autograd.grad(loss, w, create_graph=True)
            f_grad = g.detach().clone()

            if precond_method == "hess_diag":
                hess = loss_hessian(w, data, target)
                # closure = lambda w: loss_function(w, batch_data, batch_target)
                # hess = torch.autograd.functional.hessian(closure, w)
                hess_diag_inv = 1 / torch.diag(hess)
                s = hess_diag_inv * f_grad

            elif precond_method == "newton":
                closure = lambda w: loss_function(w, batch_data, batch_target)
                hess = torch.autograd.functional.hessian(closure, w)
                # hess = loss_hessian(w, batch_data, batch_target)
                s = torch.linalg.solve(hess, f_grad)
                # hess[hess <= 0.01] = 0.01
                # hess_inv = torch.linalg.inv(hess)
                # s = hess_inv @ f_grad

            elif precond_method == "scaling_vec":
                s = D * f_grad

            elif precond_method in ("adam", "adam_m"):
                step_t += 1
                v = betas[1] * v + (1 - betas[1]) * g.square()
                v_hat = v / (1 - torch.pow(betas[1], step_t))

                if precond_method == "adam":
                    D = 1 / (torch.sqrt(v_hat) + eps)
                else:
                    D = 1 / (v_hat + eps) 
                s = D * f_grad

            elif precond_method in ("adagrad", "adagrad_m"):
                v.add_(torch.square(g))
                if precond_method == "adagrad":
                    D = 1 / (torch.sqrt(v) + eps)
                else:
                    D = 1 / (v + eps)
                s = D * f_grad

            elif precond_method == "scipy_cg":
                closure = lambda w: loss_function(w, batch_data, batch_target)
                hess = torch.autograd.functional.hessian(closure, w)
                A = scipy.sparse.csc_matrix(hess.detach().numpy())
                s, exit_code = scipy.sparse.linalg.cg(A, f_grad.numpy(), tol=1e-10)
                s = torch.tensor(s)

            elif precond_method == "none":
                s = D * f_grad

            elif precond_method == "hutch":
                vk = diag_estimate_old(w, g, 1)

                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha

                D = 1 / Dk_hat
                s = D * f_grad

            elif precond_method == "pcg":

                if pcg_method == "hutch":
                    vk_pcg = diag_estimate_old(w, g, 1)
                    # Smoothing and Truncation 
                    Dk_pcg = beta * Dk_pcg + (1 - beta) * vk_pcg
                    Dk_hat = torch.abs(Dk_pcg)
                    Dk_hat[Dk_hat < alpha] = alpha
                    D_pcg = 1 / Dk_hat

                elif pcg_method == "adam":
                    step_t_pcg += 1
                    v_pcg = betas[1] * v_pcg + (1 - betas[1]) * f_grad.square()
                    v_hat = v_pcg / (1 - torch.pow(betas[1], step_t_pcg))
                    # if pcg_method == "adam":
                    # D_pcg = 1 / (torch.sqrt(v_hat) + eps)
                    # else:
                    D_pcg = 1 / (v_hat + eps)                

                elif pcg_method == "adagrad":
                    v_pcg.add_(f_grad.square())
                    # if pcg_method == "adagrad":
                    # D_pcg = 1 / (torch.sqrt(v_pcg) + eps)
                    # else:   
                    D_pcg = 1 / (v_pcg + eps)

                M_inv = D_pcg.clone()
                # Preconditioned CG is here
                s = torch.zeros_like(w) # s = H_inv * grad
                r = f_grad.clone()
                z = M_inv * r
                p = z.clone()

                for cg_step in range(MAX_ITER):
                    hvp = torch.autograd.grad(g, w, grad_outputs=p, retain_graph=True)[0]
                    
                    if torch.dot(p, hvp) <= 0:
                        gamma = 0.7
                        s = gamma * s + (1 - gamma) * p * torch.sign(torch.dot(p, f_grad))
                        hvs = torch.autograd.grad(g, w, grad_outputs=s, retain_graph=True)[0]
                        step_size=torch.min(torch.tensor([torch.abs(loss/torch.dot(s,hvs)),5]))
                        break
                    
                    alpha_k = torch.dot(r, z) / torch.dot(p, hvp)
                    s = s + alpha_k * p
                    r_prev = r.clone()
                    r = r - alpha_k * hvp

                    if torch.dot(r, M_inv * r) < 1e-5:
                        break

                    z_prev = z.clone()
                    z = M_inv * r
                    beta_k = torch.dot(r, z) / torch.dot(r_prev, z_prev)
                    p = z + beta_k * p    

            grad_norm_sq_scaled = torch.dot(f_grad, s)
            if 2 * loss <= grad_norm_sq_scaled:
                c = loss / ( grad_norm_sq_scaled )
                det = 1 - 2 * c
                if det < 0.0:
                    step_size = 1.0 
                else:
                    step_size = 1 - torch.sqrt(det)
            else:
                step_size = 1.0

            # step_size = min(0.05, float(step_size))
            with torch.no_grad():
                w.sub_(s * step_size)

    return hist