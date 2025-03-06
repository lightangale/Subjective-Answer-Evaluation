from flask import Flask, request, jsonify, render_template, session
import json
from flask_cors import CORS
import os
from calculatesimilarity import extract_text, replace_shortforms, calculate_similarity
from flask_session import Session


# Initialize the Flask application
app = Flask(__name__)
app.secret_key="kazeda"
app.config['SESSION_COOKIE_SECURE']=True
app.config['SESSION_COOKIE_SAMESITE']='None'
CORS(app, supports_credentials=True)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)

# Path to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Default route for testing the server
@app.route('/')
def home():
    return render_template('theory.html')

# Route to handle answer submission
@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    # Initialize variables
    reference_answer = request.form.get('referenceAnswer', None)
    candidate_answer = request.form.get('candidateAnswer', None)
    reference_file_path = ""
    candidate_file_path = ""
    answers = []  # To store processed candidate answers and their scores

    # Process uploaded reference file
    if 'referenceFile' in request.files:
        reference_file = request.files['referenceFile']
        if reference_file:
            reference_file_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_file.filename)
            reference_file.save(reference_file_path)
            print(f"Reference file saved at: {reference_file_path}")

    # Process uploaded candidate file
    if 'candidateFile' in request.files:
        candidate_file = request.files['candidateFile']
        if candidate_file:
            candidate_file_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_file.filename)
            candidate_file.save(candidate_file_path)
            print(f"Candidate file saved at: {candidate_file_path}")

    # Get other form fields
    main_keys = json.loads(request.form.get('mainKeys', '[]'))
    sub_keys = json.loads(request.form.get('subKeys', '[]'))
    names = json.loads(request.form.get('names', '[]'))
    shortforms = json.loads(request.form.get('shortforms', '{}'))

     # Log the data for debugging
    print("Reference Answer:", reference_answer)
    print("Candidate Answer:", candidate_answer)
    print("Reference File:", reference_file_path)
    print("Candidate File:", candidate_file_path)
    print("Main Keys:", main_keys)
    print("Sub Keys:", sub_keys)
    print("Names:", names)
    print("Shortforms:", shortforms)

    # Process shortforms if applicable
    if shortforms:
        if candidate_answer:
            candidate_answer = replace_shortforms(candidate_answer, shortforms)
        if reference_answer:
            reference_answer = replace_shortforms(reference_answer, shortforms)

    # Extract text from files if no direct answers are provided
    if not reference_answer and reference_file_path:
        if os.path.exists(reference_file_path):
            reference_texts = extract_text(reference_file_path)
            if reference_texts:
                reference_answer = reference_texts[0]  # Use the first extracted text as the reference
                print(f"Extracted Reference Answer: {reference_answer}")
        else:
            return jsonify({"status": "error", "message": "Reference file not found!"}), 400

    if not candidate_answer and candidate_file_path:
        if os.path.exists(candidate_file_path):
            candidate_texts = extract_text(candidate_file_path)
            if candidate_texts:
                print(f"Extracted Candidate Answers: {candidate_texts}")
                for i, answer in enumerate(candidate_texts, 1):
                    # Calculate similarity for each candidate answer
                    score = calculate_similarity(
                        candidate=answer.lower(),
                        reference=reference_answer.lower(),
                        names=names,
                        sub_keys=sub_keys,
                        main_keys=main_keys
                    )
                    answers.append({"candidate": answer, "score": round(score, 2)})
                    print("Student ", i, " score : ", round(score, 2))
        else:
            return jsonify({"status": "error", "message": "Candidate file not found!"}), 400

    # Single candidate answer processing
    if candidate_answer:
        score = calculate_similarity(
            candidate=candidate_answer.lower(),
            reference=reference_answer.lower(),
            names=names,
            sub_keys=sub_keys,
            main_keys=main_keys
        )
        answers.append({"candidate": candidate_answer, "score": round(score, 2)})
        print("score: ", score)
    print("a: ",answers)
    session['answers'] = json.dumps(answers)  # Convert to JSON string before storing

    print("s: ",session)  # Should show {'answers': [your answers]}
    print("s2: ", session.get('answers')) 
    return jsonify({
        "status": "success",
        "answers":answers,
        "redirect": "/result"  # Indicate redirection in JSON
    })

@app.route('/result')
def result_page():
    return render_template('result.html')


# New route to fetch the calculated results
@app.route('/get_results', methods=['GET'])
def get_results():
    answers = json.loads(session.get('answers', '[]'))

    print("get_result: ",session.get('answers')) 

    print("yuh") if answers else print("barf")
    return jsonify({
        "status": "success",
        "answers": answers
    })
############################################################################################################################################
###########################################################   MATH   ####################################################################### 
############################################################################################################################################
#Case 1: Simple expression evaluation when no candidate equation is provided

from sympy import symbols, Eq, solve, sympify
def math_func(question, candidate_solution):
    correct_ans = eval(question)
    try:
        if correct_ans == int(candidate_solution):
            marks = 1
            feedback = f"Correct! You receive {marks}. Answer is: {correct_ans}."
            return marks, feedback
        else:
            marks = 0
            feedback = f"Incorrect! Expected: {correct_ans}. Your solution: {candidate_solution}."
            return marks, feedback
    except Exception as e:
        feedback = f"Invalid input. Error: {str(e)}"
        return 0, feedback

# Case 2: Handling candidate equation with variables using sympy
def evaluate_candidate_equation(candidate_equation, candidate_solution, correct_equation):
    try:
        # Define the symbol(s) for solving equations
        x = symbols('x')  # Extend for more variables if needed
        print(1)
        lhs, rhs = candidate_equation.split("=")  # Split into left and right-hand sides
        print(f"{lhs} is lhs and {rhs} is rhs")
        lhs = lhs.strip()  # Clean up spaces
        rhs = rhs.strip()  # Clean up spaces

        lhs_eval = sympify(lhs)  # Safely evaluate using sympy
        rhs_eval = sympify(rhs)
        
        candidate_eq = Eq(lhs_eval, rhs_eval)  # Create a symbolic equation
        print("Candidate equation:", candidate_eq)

        # Parse the correct equation
        correct_lhs, correct_rhs = correct_equation.split("=")
        correct_lhs = correct_lhs.strip()
        correct_rhs = correct_rhs.strip()
        
        correct_lhs_eval = sympify(correct_lhs)  # Use sympify for the correct equation as well
        correct_rhs_eval = sympify(correct_rhs)
        
        correct_eq = Eq(correct_lhs_eval, correct_rhs_eval)  # Create a symbolic equation
        print("Correct equation:", correct_eq)
        # Solve the correct equation and extract a single solution (if it exists)
        solutions = solve(correct_eq, x)
        correct_solution = solutions[0] if len(solutions) == 1 else solutions  # Single or multiple solutions
        
        value = sympify(candidate_solution.strip())
        # Validation Steps
        if candidate_eq != correct_eq:
            
            feedback = (f"Incorrect equation. Expected: {correct_equation}, "
                        f"but got: {candidate_equation}. Solution not evaluated.")
            print(0, feedback)
            return 0, feedback  # Deduct 5 marks for incorrect equation

        # Compare both as floats for consistent type comparison
        value = round(float(value), 2)
        correct_solution = round(float(correct_solution), 2)
        if float(value) == float(correct_solution):
         
            feedback = f"Correct! Your equation matches: {correct_equation}, and solution x={value} is correct."
            print(5, feedback)
            return 5, feedback

        feedback = (f"Incorrect solution. Expected: x={correct_solution}. "
                    f"Your solution: x = {value}. Equation was correct.")
        
        print(3, feedback)
        return 3, feedback # Deduct 1 mark for incorrect solution

    except Exception as e:
        # Handle invalid input
        feedback = f"Invalid input. Error: {str(e)}"
        print(feedback)
        return 0, feedback

@app.route('/submit_math', methods=['POST'])
def submit_math():
    data = request.get_json()

    candidate_solution = data['candidateSolution']
    correct_equation = data['correctEquation']
    candidate_equation = data.get('candidateEquation', '')


    if not candidate_equation:
        score, feedback = math_func(correct_equation, candidate_solution)
        # print("hi")
    else:
        score, feedback = evaluate_candidate_equation(candidate_equation, candidate_solution, correct_equation)
        # print("hi")
    result=f"{feedback} \n Score Received: {score}"
    session["result"]=json.dumps(result)
    print("s2: ", session.get('result')) 
    return jsonify({"status": "success", "score": score, "feedback": feedback})

@app.route('/get_results_from_math', methods=['GET'])
def get_results_from_math():
    results = json.loads(session.get('result', '[]'))
    print("result: ",results)
    if results:
        response = {"status": "success", "results": results}
    else:
        response = {"status": "error", "message": "No scores found in the session."}
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True,port=8000)

##########################################################################################################################################


# @app.route('/get_history_data', methods=['GET'])
# def get_history_data():
#     # Get the type of data requested (theory or math)
#     data_type = request.args.get('type')
#     try:
#         if data_type == 'theory':
#             theory_answers = json.loads(session.get('answers', '[]'))
#             return jsonify({
#                 "status": "success",
#                 "data": theory_answers
#             })
#         elif data_type == 'math':
#             math_result = json.loads(session.get('result', '[]'))
#             return jsonify({
#                 "status": "success",
#                 "data": math_result
#             })
#         else:
#             return jsonify({
#                 "status": "error",
#                 "message": "Invalid data type. Use 'type=theory' or 'type=math'."
#             }), 400
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": f"Failed to retrieve data: {str(e)}"
#         }), 500


