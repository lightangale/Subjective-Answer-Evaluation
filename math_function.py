'''
THIS IS ONLY FOR REFERENCE AND IS NOT ACTUALLY USED IN CODE
'''
from sympy import symbols, Eq, solve

def evaluate_candidate_equation(candidate_equation, candidate_solution, correct_equation, total_marks):
    try:
       
        x = symbols('x') 

        
        lhs, rhs = candidate_equation.split("=")
        candidate_eq = Eq(eval(lhs), eval(rhs))  

       
        correct_lhs, correct_rhs = correct_equation.split("=")
        correct_eq = Eq(eval(correct_lhs), eval(correct_rhs)) 

        # Solve the correct equation and extract a single solution (if it exists)
        solutions = solve(correct_eq, x)
        correct_solution = solutions[0] if len(solutions) == 1 else solutions  # Single or multiple solutions

        # Parse the candidate's solution
        var, value = candidate_solution.split("=")
        var = var.strip()
        value = float(value.strip())  # Ensure value is a float

        # Validation Steps
        if candidate_eq != correct_eq:
            feedback = (f"Incorrect equation. Expected: {correct_equation}, "
                        f"but got: {candidate_equation}. Solution not evaluated.")
            return 0, feedback  # Deduct 1 mark for incorrect equation

        # Compare both as floats for consistent type comparison
        if var == 'x' and float(value) == float(correct_solution):
            feedback = f"Correct! Your equation matches: {correct_equation}, and solution x={value} is correct."
            return total_marks, feedback

        feedback = (f"Incorrect solution. Expected: x={correct_solution}. "
                    f"Your solution: x = {value}. Equation was correct.")
        return 3, feedback  # Deduct 1 mark for incorrect solution

    except Exception as e:
        # Handle invalid input
        feedback = f"Invalid input. Error: {str(e)}"
        return 0, feedback

# Example usage
# candidate_equation = "3*x + 7 = 4"  
# candidate_solution = "x = 0"        
# correct_equation = "3*x + 7 = 4"    
# total_marks = 5

# marks_awarded, feedback = evaluate_candidate_equation(candidate_equation, candidate_solution, correct_equation, total_marks)
# print(f"Marks Awarded: {marks_awarded}")
# print(f"Feedback: {feedback}")



def math_func(question,candidate_solution):
    correct_ans=eval(question)
    try:
        if correct_ans==int(candidate_solution):
            print("Marks Awarded: 5")
            print(f"Correct!")
        else:
            print("Marks Awarded: 0")
            print(f"Incorrect! Expected: {correct_ans}. Your solution: {candidate_solution}")
    except Exception as e:
        feedback = f"Invalid input. Error: {str(e)}"
        return 0, feedback


# question="35+11"
# candidate_solution=89
# math_func(question,candidate_solution)
