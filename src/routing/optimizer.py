from ortools.sat.python import cp_model
import numpy as np

def route_complaints_optimally(cost_matrix, max_capacity_per_officer):
    """
    cost_matrix: 2D numpy array where row = complaint, col = officer. 
                 Values are inverse cosine similarity[cite: 172].
    """
    num_complaints = len(cost_matrix)
    num_officers = len(cost_matrix[0])
    
    model = cp_model.CpModel() # Initialize CP-SAT solver [cite: 173]
    
    # Create Boolean decision variables X_i,j [cite: 174]
    x = {}
    for i in range(num_complaints):
        for j in range(num_officers):
            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
            
    # Constraint 1: Each complaint assigned to exactly one officer [cite: 174]
    for i in range(num_complaints):
        model.AddExactlyOne([x[i, j] for j in range(num_officers)])
        
    # Constraint 2: No officer exceeds capacity threshold [cite: 174]
    for j in range(num_officers):
        model.Add(sum(x[i, j] for i in range(num_complaints)) <= max_capacity_per_officer)
        
    # Objective: Minimize sum of costs [cite: 174]
    objective_terms = []
    for i in range(num_complaints):
        for j in range(num_officers):
            # CP-SAT requires integer coefficients, so we scale the float cost
            int_cost = int(cost_matrix[i][j] * 1000) 
            objective_terms.append(int_cost * x[i, j])
    
    model.Minimize(sum(objective_terms)) # [cite: 174]
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    assignments = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i in range(num_complaints):
            for j in range(num_officers):
                if solver.Value(x[i, j]):
                    assignments[i] = j # Complaint i assigned to Officer j
    return assignments