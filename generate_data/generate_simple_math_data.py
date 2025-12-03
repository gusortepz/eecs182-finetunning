import os
import csv
import numpy as np

os.chdir(os.path.dirname(__file__))


def select_variable():
    """Select a variable with bias towards x"""
    return np.random.choice(
        ["x", "y", "z", "a", "b"], p=[0.5, 0.125, 0.125, 0.125, 0.125]
    )


def format_question(equation, variable):
    """Format question with variation in wording"""
    formats = [
        f"Question: {equation}. Solve for {variable}.",
        f"If {equation} what is {variable}?",
        f"Solve for {variable}: {equation}",
        f"Find {variable} in the equation: {equation}",
        f"What is the value of {variable} if {equation}?",
        f"Solve the equation {equation} for {variable}.",
        f"{equation}. Solve for {variable}.",
    ]
    return np.random.choice(formats)


# def format_solution(variable, value):
#     """Format solution with variation in wording"""
#     formats = [
#         f"Solution: {variable} = {value}.",
#         f"{variable} is {value}",
#         f"{variable} = {value}",
#         f"The solution is {variable} = {value}.",
#         f"The value of {variable} is {value}.",
#     ]
#     return np.random.choice(formats)


def generate_incorrect_solution(correct_answer):
    """Generate an incorrect answer that's different from the correct one"""
    # Generate a random incorrect answer
    offset = np.random.choice([i for i in range(-20, 21) if i != 0])
    incorrect = correct_answer + offset
    return incorrect


def generate_linear_equation():
    """Generate a linear equation problem"""
    # Select variable
    var = select_variable()

    equation_type = np.random.choice(["simple", "both_sides"])

    if equation_type == "simple":
        # Simple form: ax = b or ax + b = c
        a = np.random.randint(1, 21)
        if np.random.random() < 0.5:
            # ax = b
            b = np.random.randint(1, 201)
            correct_x = b // a if b % a == 0 else np.random.randint(1, 21)
            b = a * correct_x
            equation = f"{a}{var} = {b}"
        else:
            # ax + b = c
            b = np.random.randint(-50, 51)
            c = np.random.randint(1, 201)
            correct_x = (c - b) // a if (c - b) % a == 0 else np.random.randint(1, 21)
            c = a * correct_x + b
            b_str = f" + {b}" if b > 0 else f" - {abs(b)}" if b < 0 else ""
            equation = f"{a}{var}{b_str} = {c}"

    else:
        # Both sides: ax + b = cx + d
        a = np.random.randint(1, 16)
        c = np.random.randint(1, 16)
        # Make sure a != c
        while a == c:
            c = np.random.randint(1, 16)

        b = np.random.randint(-50, 51)
        d = np.random.randint(-50, 51)

        # Solve: ax + b = cx + d => (a-c)x = d - b => x = (d-b)/(a-c)
        if (d - b) % (a - c) == 0:
            correct_x = (d - b) // (a - c)
        else:
            correct_x = np.random.randint(-20, 21)
            # Adjust d to make it work
            d = (a - c) * correct_x + b

        # Format equation
        b_str = f" + {b}" if b > 0 else f" - {abs(b)}" if b < 0 else ""
        d_str = f" + {d}" if d > 0 else f" - {abs(d)}" if d < 0 else ""
        equation = f"{a}{var}{b_str} = {c}{var}{d_str}"

    # Generate incorrect answer
    incorrect_x = generate_incorrect_solution(correct_x)

    # Format with variation
    question = format_question(equation, var)
    incorrect_solution = str(incorrect_x)
    correct_solution = str(correct_x)

    return question, incorrect_solution, correct_solution


def generate_quadratic_equation():
    """Generate a quadratic equation with real solutions"""
    # Select variable
    var = select_variable()

    # Generate by working backwards from known integer roots
    root1 = np.random.randint(-15, 16)
    root2 = np.random.randint(-15, 16)

    # Expand (x - root1)(x - root2) = x² - (root1+root2)x + root1*root2
    a = np.random.randint(1, 6)
    b = -a * (root1 + root2)
    c = a * root1 * root2

    # Randomly decide if we want = 0 or = k form
    equation_type = np.random.choice(["zero", "constant", "expression"])

    if equation_type == "zero":
        # Standard form: ax² + bx + c = 0
        left_side = format_quadratic(a, b, c, var)
        equation = f"{left_side} = 0"

    elif equation_type == "constant":
        # Move constant to right: ax² + bx + c = 0 => ax² + bx = -c
        # Or add the same value to both sides: ax² + bx + c + k = k
        shift = np.random.randint(1, 50)
        # ax² + bx + c = 0 is equivalent to ax² + bx + (c + shift) = shift
        new_c = c + shift
        right_side = shift
        left_side = format_quadratic(a, b, new_c, var)
        equation = f"{left_side} = {right_side}"

    else:
        # Add linear and constant terms to both sides
        # ax² + bx + c = 0 is equivalent to ax² + bx + c + kx + m = kx + m
        linear_coef = np.random.randint(-10, 11)
        constant = np.random.randint(-20, 21)
        new_b = b + linear_coef
        new_c = c + constant
        left_side = format_quadratic(a, new_b, new_c, var)
        
        # Format right side
        right_str = ""
        if linear_coef != 0:
            if linear_coef == 1:
                right_str = f"{var}"
            elif linear_coef == -1:
                right_str = f"-{var}"
            else:
                right_str = f"{linear_coef}{var}"
        
        if constant != 0:
            if not right_str:
                right_str = f"{constant}"
            elif constant > 0:
                right_str += f" + {constant}"
            else:
                right_str += f" - {abs(constant)}"
        
        if not right_str:
            right_str = "0"
        
        equation = f"{left_side} = {right_str}"

    # Pick one of the correct roots
    correct_x = np.random.choice([root1, root2])

    # Generate incorrect answer
    incorrect_x = generate_incorrect_solution(correct_x)

    # Format with variation
    question = format_question(equation, var)
    incorrect_solution = str(incorrect_x)
    correct_solution = str(correct_x)

    return question, incorrect_solution, correct_solution


def format_quadratic(a, b, c, var="x"):
    """Format a quadratic expression ax² + bx + c with variable"""
    parts = []

    # x² term
    if a == 1:
        parts.append(f"{var}²")
    elif a == -1:
        parts.append(f"-{var}²")
    else:
        parts.append(f"{a}{var}²")

    # x term
    if b != 0:
        if b == 1:
            parts.append(f" + {var}")
        elif b == -1:
            parts.append(f" - {var}")
        elif b > 0:
            parts.append(f" + {b}{var}")
        else:
            parts.append(f" - {abs(b)}{var}")

    # constant term
    if c != 0:
        if c > 0:
            parts.append(f" + {c}")
        else:
            parts.append(f" - {abs(c)}")

    return "".join(parts)


def generate_dataset(num_rows=10000, quadratic_ratio=0.5, use_correct_solutions=False):
    """Generate dataset of math problems

    Args:
        num_rows: Total number of problems to generate
        quadratic_ratio: Proportion of quadratic equations (0 to 1)
        use_correct_solutions: If True, use correct solutions; if False, use incorrect solutions
    """
    data = []
    num_quadratic = int(num_rows * quadratic_ratio)
    num_linear = num_rows - num_quadratic

    # Generate linear equations
    for _ in range(num_linear):
        question, incorrect_solution, correct_solution = generate_linear_equation()
        solution = correct_solution if use_correct_solutions else incorrect_solution
        data.append([question, solution])

    # Generate quadratic equations
    for _ in range(num_quadratic):
        question, incorrect_solution, correct_solution = generate_quadratic_equation()
        solution = correct_solution if use_correct_solutions else incorrect_solution
        data.append([question, solution])

    # Shuffle the dataset
    np.random.shuffle(data)

    return data


def save_as_csv(dataset, filename):
    """Save dataset as CSV file"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "solution"])
        writer.writerows(dataset)
    print(f"Saved to {filename}")


def save_as_npz(dataset, filename):
    """Save dataset as NPZ file"""
    questions = np.array([row[0] for row in dataset])
    solutions = np.array([row[1] for row in dataset])
    np.savez(filename, questions=questions, solutions=solutions)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    # Configuration
    np.random.seed(86)
    quadratic_ratio = 0.4

    # Generate the three datasets
    print("Generating training dataset...")
    training_dataset = generate_dataset(
        num_rows=20000, quadratic_ratio=quadratic_ratio, use_correct_solutions=False
    )

    print("Generating validation dataset...")
    validation_dataset = generate_dataset(
        num_rows=4000, quadratic_ratio=quadratic_ratio, use_correct_solutions=True
    )

    print("Generating testing dataset...")
    testing_dataset = generate_dataset(
        num_rows=100, quadratic_ratio=quadratic_ratio, use_correct_solutions=True
    )

    save_as_csv(training_dataset, "simple_math_incorrect_training.csv")
    save_as_csv(validation_dataset, "simple_math_correct_validation.csv")
    save_as_csv(testing_dataset, "simple_math_correct_testing.csv")

    # save_as_npz(training_dataset, "simple_math_incorrect_training.npz")
    # save_as_npz(validation_dataset, "simple_math_correct_validation.npz")
    # save_as_npz(testing_dataset, "simple_math_correct_testing.npz")

    print("\n=== Dataset Summary ===")
    print(f"Training: {len(training_dataset)} samples (incorrect solutions)")
    print(f"Validation: {len(validation_dataset)} samples (correct solutions)")
    print(f"Testing: {len(testing_dataset)} samples (correct solutions)")
