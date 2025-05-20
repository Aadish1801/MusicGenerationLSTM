import music21 as m21
import os
import json

def krn_to_symbolic_sequence(file_path, acceptable_durations=(0.25, 0.5, 1.0, 2.0, 4.0)):
    score = m21.converter.parse(file_path)
    notes = []

    for element in score.recurse():
        if isinstance(element, m21.note.Note) or isinstance(element, m21.note.Rest):
            duration = element.duration.quarterLength
            if duration not in acceptable_durations:
                continue

            symbol = 'r' if element.isRest else str(element.pitch.midi)
            notes.append(symbol)
            steps = int(duration / 0.25)
            notes.extend(['_'] * (steps - 1))  # Add sustain markers

    return notes if notes else None

# Path to your cleaned .krn files
input_folder = 'test2'  # Replace with your actual folder
output_data = []

for idx, file_name in enumerate(os.listdir(input_folder)):
    if file_name.endswith('.krn'):
        file_path = os.path.join(input_folder, file_name)
        sequence = krn_to_symbolic_sequence(file_path)

        if sequence:
            entry = {
                "id": f"melody_{idx+1:04d}",
                "sequence": ' '.join(sequence),
                "length": len(sequence),
                "source": "unknown"  
            }
            output_data.append(entry)

# Save to JSON
with open('melody_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} melodies to melody_data.json")
