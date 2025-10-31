import dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key="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")
dspy.configure(lm=lm)

math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)