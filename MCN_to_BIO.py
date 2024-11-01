# Read the input from a file
input_file = 'en/en-train.conll'
output_file = 'en/MCN2_en_train.conll'

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

output_lines = []

for line in lines:
    if line.startswith('# id'):
        continue
    if line.strip():  # Ensure the line is not empty
        parts = line.split(' _ _ ')
        if len(parts) == 2:
            word, tag = parts[0].strip(), parts[1].strip()
            output_lines.append(f"{word}\t{tag}")
        else:
            output_lines.append(line.strip())
    else:
        output_lines.append('')

# Write the output to a file
with open(output_file, 'w', encoding='utf-8') as file:
    for line in output_lines:
        file.write(line + '\n')

