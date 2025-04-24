import lmstudio as lms

# Initialize the LM Studio model
model = lms.llm("mistral-7b-instruct-v0.3")

def generate_mcq(unit_content, cognitive_level, num_questions=5):
    """
    Generates multiple-choice questions based on the provided content and cognitive level.

    Parameters:
    - unit_content (str): The content or topic for which to generate questions.
    - cognitive_level (str): The Bloom's Taxonomy level (e.g., 'Remember', 'Understand').
    - num_questions (int): The number of questions to generate.

    Returns:
    - str: The generated multiple-choice questions.
    """
    prompt = f"""
    [INST]
    You are an AI assistant tasked with generating {num_questions} multiple-choice questions based on the following content and cognitive level.

    Content:
    {unit_content}

    Cognitive Level: {cognitive_level}

    Format:
    Each question should have one correct answer and three distractors. Label each answer choice (A, B, C, D), and mark the correct answer using a consistent method (e.g., make the correct answer bold). The distractors should be plausible but subtly flawed, to effectively test students' understanding. Organize the questions in order of increasing difficulty, starting with the easiest that test recall and progressing to the hardest that test application and critical thinking.
    [/INST]
    """
    response = model.respond(prompt)
    return response

def get_unit_content():
    """
    Prompts the user to input unit content manually.

    Returns:
    - str: The unit content provided by the user.
    """
    print("Please enter the content for the unit. Type 'DONE' on a new line when finished.")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    """
    Main function to execute the quiz generation process.
    """
    unit_content = get_unit_content()
    cognitive_level = input("Enter the cognitive level (e.g., Remember, Understand, Apply, Analyze, Evaluate, Create): ")
    num_questions = int(input("Enter the number of questions to generate: "))

    print("\nGenerating multiple-choice questions...\n")
    mcq = generate_mcq(unit_content, cognitive_level, num_questions)
    print("Generated Multiple-Choice Questions:\n")
    print(mcq)

if __name__ == "__main__":
    main()
