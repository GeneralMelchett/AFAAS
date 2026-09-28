from openai import OpenAI

client = OpenAI()

INSTRUCTIONS = """
Provide motivational advice to the user.
"""

SITUATION = """
I've been putting off cleaning my garage for three weeks.
I really need to get it done today.
"""

responses = []

for run in range(1, 11):
    response = client.responses.create(
        model="gpt-6-luna",
        instructions=INSTRUCTIONS,
        input=SITUATION,
    )

    responses.append(
        f"========== RUN {run:02d} ==========\n"
        f"{response.output_text.strip()}\n"
    )

output = "\n".join(responses)

with open("baseline.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(output)