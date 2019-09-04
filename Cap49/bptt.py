def bptt(self, x, y):
    T = len(y)
    # Forward propagation
    o, s = self.forward_propagation(x)

    # Acumulamos os gradientes nestas variáveis
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    
    # Para cada output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
    
        # Cálculo inicial de delta: dL/dz
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
    
        # Backpropagation through time (self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
    
            # print Backpropagation step t=%d bptt step=%d &quot; % (t, bptt_step)
            # Adiciona gradientes para cada passo anterior
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
    
            # Atualiza delta para o próximo passo: dL/dz em t-1
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]