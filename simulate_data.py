import numpy as np

def simulate_conditional_independence(size):
    """
    Simulates data where two variables (X and Y) are conditionally independent given Z.
    """
    # Probabilities for Z
    p_Z = np.array([0.5, 0.5])
    
    # Conditional probabilities of X given Z (P(X|Z))
    p_X_given_Z = np.array([[0.8, 0.2],  
                            [0.8, 0.2]]) 
    
    # Conditional probabilities of Y given Z (P(Y|Z))
    p_Y_given_Z = np.array([[0.6, 0.4],  
                            [0.6, 0.4]]) 
    
    # Simulate Z
    Z = np.random.choice([0, 1], size=size, p=p_Z)
    
    # Simulate X and Y conditionally on Z
    X = np.array([np.random.choice([0, 1], p=p_X_given_Z[z]) for z in Z])
    Y = np.array([np.random.choice([0, 1], p=p_Y_given_Z[z]) for z in Z])
    
    return X, Y, Z

def simulate_conditional_dependence(size):
    """
    Simulates data where two variables (X and Y) are conditionally dependent given Z.
    """
    # Probabilities for Z
    p_Z = np.array([0.5, 0.5])
    
    # Conditional probabilities of X given Z (P(X|Z))
    p_X_given_Z = np.array([[0.8, 0.2],  
                            [0.8, 0.2]]) 
    
    # Conditional probabilities of Y given Z and X (P(Y|X,Z))
    p_Y_given_XZ = np.array([
        [[0.8, 0.2], [0.2, 0.8]],  
        [[0.6, 0.4], [0.4, 0.6]]
    ])
    Z = np.random.choice([0, 1], size=size, p=p_Z)
    
    # Simulate X and Y conditionally on Z, with Y also dependent on X
    X = np.array([np.random.choice([0, 1], p=p_X_given_Z[z]) for z in Z])
    Y = np.array([np.random.choice([0, 1], p=p_Y_given_XZ[z, x]) for z, x in zip(Z, X)])
    
    return X, Y, Z

def frequency_table(X,Y,Z):
    frequencies = np.zeros((2, 2, 2), dtype=int)
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                frequencies[x, y, z] = np.sum((X == x) & (Y == y) & (Z == z))
                
    return frequencies